import argparse
import csv
import torch
import torch.nn as nn
import torch.utils.benchmark as benchmark
import sys

DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

_DEFAULT_SIZE=2560
def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch GPU benchmark (torch.utils.benchmark)")
    parser.add_argument(
        "--dtype",
        type=str,
        default="all",
        choices=list(DTYPE_MAP.keys()) + ["all"],
        help="Data type: fp32 | fp16 | bf16 | all",
    )
    parser.add_argument("--matmul-size", type=int, default=_DEFAULT_SIZE,
                        help="Square matrix size for GEMM benchmark (N x N).")
    parser.add_argument("--batch-size", type=int, default=_DEFAULT_SIZE,
                        help="Batch size for training step benchmark.")
    parser.add_argument("--min-run-time", type=float, default=30.0,
                        help="Minimum run time (in seconds) for blocked_autorange.")
    parser.add_argument("--conv-size", type=int, default=1024,
                        help="Input image size for Conv2d benchmark (HxW).")
    parser.add_argument("--elementwise-size", type=int, default=128 * _DEFAULT_SIZE * _DEFAULT_SIZE,
                        help="Number of elements for elementwise benchmark.")
    parser.add_argument("--model-dim", type=int, default=_DEFAULT_SIZE,
                        help="Input feature dimension for MLP benchmark.")
    parser.add_argument("--mem-bandwidth-size", type=int, default=256 * _DEFAULT_SIZE * _DEFAULT_SIZE,
                        help="Number of elements for memory bandwidth benchmark (y.copy_(x)).")
    parser.add_argument("--output", type=str, default="benchmark_raw_times.csv",
                        help="Output CSV filename for all raw iteration times.")
    # torchbench extras
    parser.add_argument("--disable-torchbench", action="store_true",
                        help="Disables TorchBench")
    parser.add_argument("--torchbench-models", type=str,
                        default="resnet50,BERT_pytorch,nanogpt",
                        help="Comma-separated TorchBench model names to run (if available).")
    parser.add_argument(
        "--torchbench-mode",
        type=str,
        default="both",
        choices=["train", "eval", "both"],
        help="Run TorchBench models in train, eval or both modes.",
    )
    return parser.parse_args()


def has_cuda():
    return torch.cuda.is_available()


def timer_stmt(stmt, globals=None, label="", sub_label="", description=""):
    return benchmark.Timer(
        stmt=stmt,
        globals=globals or {},
        label=label,
        sub_label=sub_label,
        description=description,
        num_threads=1,
    )


def run_with_memory(timer_obj, min_run_time):
    peak_alloc = 0
    peak_reserved = 0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        res = timer_obj.blocked_autorange(min_run_time=min_run_time)

        torch.cuda.synchronize()
        peak_alloc = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()

        torch.cuda.empty_cache()
    else:
        res = timer_obj.blocked_autorange(min_run_time=min_run_time)
    return res, peak_alloc, peak_reserved


def benchmark_matmul(device, size, dtype, min_run_time):
    x = torch.randn(size, size, device=device, dtype=dtype)
    y = torch.randn(size, size, device=device, dtype=dtype)
    t = timer_stmt("x @ y", globals={"x": x, "y": y},
                   label="GEMM", sub_label=f"{size}x{size}",
                   description=f"{device}, {dtype}")
    return run_with_memory(t, min_run_time)


def benchmark_conv(device, conv_size, dtype, min_run_time):
    model = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1).to(device=device, dtype=dtype)
    inp = torch.randn(16, 64, conv_size, conv_size, device=device, dtype=dtype)
    t = timer_stmt("model(inp)", globals={"model": model, "inp": inp},
                   label="Conv2d", sub_label=f"16x64x{conv_size}x{conv_size}",
                   description=f"{device}, {dtype}")
    return run_with_memory(t, min_run_time)


def benchmark_elementwise(device, n, dtype, min_run_time):
    x = torch.randn(n, device=device, dtype=dtype)
    y = torch.randn(n, device=device, dtype=dtype)
    t = timer_stmt("x + y; x * y", globals={"x": x, "y": y},
                   label="Elementwise", sub_label=f"n={n:,}",
                   description=f"{device}, {dtype}")
    return run_with_memory(t, min_run_time)


def benchmark_forward_backward(device, batch, dim, dtype, min_run_time):
    model = nn.Sequential(
        nn.Linear(dim, 4096),
        nn.ReLU(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 1000),
    ).to(device=device, dtype=dtype)

    x = torch.randn(batch, dim, device=device, dtype=dtype, requires_grad=False)
    target = torch.randn(batch, 1000, device=device, dtype=dtype)
    loss_fn = nn.MSELoss()
    use_autocast = device == "cuda" and dtype in (torch.float16, torch.bfloat16)

    def fwd_bwd():
        model.zero_grad(set_to_none=True)
        if use_autocast:
            with torch.autocast(device_type="cuda", dtype=dtype):
                out = model(x)
                loss = loss_fn(out, target)
        else:
            out = model(x)
            loss = loss_fn(out, target)
        loss.backward()

    t = timer_stmt("fwd_bwd()", globals={"fwd_bwd": fwd_bwd},
                   label="Train step", sub_label=f"batch={batch}, dim={dim}",
                   description=f"{device}, {dtype}")
    return run_with_memory(t, min_run_time)


def benchmark_mem_bandwidth(device, n, dtype, min_run_time):
    x = torch.randn(n, device=device, dtype=dtype)
    y = torch.empty_like(x)
    elem_size = x.element_size()
    bytes_moved = 2 * n * elem_size
    t = timer_stmt("y.copy_(x)", globals={"x": x, "y": y},
                   label="Memory bandwidth", sub_label=f"n={n:,}",
                   description=f"{device}, {dtype}, bytes_moved={bytes_moved}")
    res, peak_alloc, peak_reserved = run_with_memory(t, min_run_time)
    return res, peak_alloc, peak_reserved, bytes_moved


def benchmark_torchbench_model(model_name, device, dtype, min_run_time, mode="train"):
    """
    Returns (result, peak_alloc, peak_reserved, model_name, mode) or None
    """
    try:
        from torchbenchmark import load_model_by_name
    except Exception as e:
        print(f"[TorchBench] torchbenchmark not available: {e}")
        return None

    tb_model = load_model_by_name(
        model_name,
    )(device=device,
      test=mode)
    if dtype=="bf16":
        tb_model.enable_bf16()
    if dtype=="fp16":
        tb_model.enable_fp16()
    def fn():
        tb_model.eval()
    
    sub_label = f"{model_name} ({mode})"

    t = timer_stmt("fn()", globals={"fn": fn},
                   label="TorchBench", sub_label=sub_label,
                   description=f"{device}, {dtype}")
    res, peak_alloc, peak_reserved = run_with_memory(t, min_run_time)
    return res, peak_alloc, peak_reserved, model_name, mode


def save_raw_times_to_csv(results_with_mem, membw_results, torchbench_results, filename, torch_version):
    """
    results_with_mem: list of tuples (res, peak_alloc, peak_reserved, dtype_str)
    membw_results: list of tuples (res, peak_alloc, peak_reserved, bytes_moved, dtype_str)
    torchbench_results: list of tuples (res, peak_alloc, peak_reserved, model_name, mode, dtype_str)
    """
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "torch_version",
            "dtype",
            "label",
            "sub_label",
            "description",
            "iteration",
            "time_sec",
            "peak_mem_alloc_bytes",
            "peak_mem_reserved_bytes",
            "bytes_moved",
            "gb_per_s",
        ])

        # built-ins
        for res, peak_alloc, peak_reserved, dtype_str in results_with_mem:
            for i, t in enumerate(res.raw_times, start=1):
                writer.writerow([
                    torch_version,
                    dtype_str,
                    res.label,
                    res.sub_label,
                    res.description,
                    i,
                    f"{t:.9f}",
                    peak_alloc,
                    peak_reserved,
                    "",
                    "",
                ])

        # membw
        for membw in membw_results:
            res, peak_alloc, peak_reserved, bytes_moved, dtype_str = membw
            for i, t in enumerate(res.raw_times, start=1):
                gbps = (bytes_moved / t) / 1e9
                writer.writerow([
                    torch_version,
                    dtype_str,
                    res.label,
                    res.sub_label,
                    res.description,
                    i,
                    f"{t:.9f}",
                    peak_alloc,
                    peak_reserved,
                    bytes_moved,
                    f"{gbps:.3f}",
                ])

        # torchbench
        for tb in torchbench_results:
            if tb is None:
                continue
            res, peak_alloc, peak_reserved, model_name, mode, dtype_str = tb
            for i, t in enumerate(res.raw_times, start=1):
                writer.writerow([
                    torch_version,
                    dtype_str,
                    res.label,
                    res.sub_label,
                    res.description,
                    i,
                    f"{t:.9f}",
                    peak_alloc,
                    peak_reserved,
                    "",
                    "",
                ])

    print(f"\n✅ Saved raw iteration times + peak mem (+ TorchBench) to {filename}")


def main():
    args = parse_args()
    torch_version = torch.__version__
    device = "cuda" if has_cuda() else "cpu"

    # decide which dtypes to run
    if args.dtype == "all":
        dtypes_to_run = ["fp32", "fp16", "bf16"]
    else:
        dtypes_to_run = [args.dtype]

    all_results = []          # for built-ins
    all_membw_results = []    # for mem bw
    all_torchbench_results = []  # for torchbench compare
    compare_inputs = []       # for benchmark.Compare

    print(f"PyTorch: {torch_version}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")
    print(
        f"Params: matmul={args.matmul_size}, conv={args.conv_size}, "
        f"elementwise={args.elementwise_size}, batch={args.batch_size}, "
        f"dim={args.model_dim}, mem_bw={args.mem_bandwidth_size}, "
        f"min_run_time={args.min_run_time}s"
    )

    for dtype_name in dtypes_to_run:
        dtype = DTYPE_MAP[dtype_name]

        print(f"\n=== Running benchmarks for dtype={dtype_name} ===")

        # TorchBench
        torchbench_results = []
        if not args.disable_torchbench:
            sys.path.append("benchmark")
            try:
                model_names = [m.strip() for m in args.torchbench_models.split(",") if m.strip()]
                for name in model_names:
                    if args.torchbench_mode in ("train", "eval"):
                        tb_res = benchmark_torchbench_model(
                            name, device, dtype, args.min_run_time, mode=args.torchbench_mode
                        )
                        if tb_res is not None:
                            # append dtype_name to tuple
                            res, pa, pr, mn, md = tb_res
                            torchbench_results.append((res, pa, pr, mn, md, dtype_name))
                            compare_inputs.append(res)
                    else:  # both
                        for mode in ("train", "eval"):
                            tb_res = benchmark_torchbench_model(
                                name, device, dtype, args.min_run_time, mode=mode
                            )
                            if tb_res is not None:
                                res, pa, pr, mn, md = tb_res
                                torchbench_results.append((res, pa, pr, mn, md, dtype_name))
                                compare_inputs.append(res)
            except Exception as e:
                print(f"[TorchBench] error while running models: {e}")

        matmul_res = benchmark_matmul(device, args.matmul_size, dtype, args.min_run_time)
        conv_res = benchmark_conv(device, args.conv_size, dtype, args.min_run_time)
        elem_res = benchmark_elementwise(device, args.elementwise_size, dtype, args.min_run_time)
        train_res = benchmark_forward_backward(device, args.batch_size, args.model_dim, dtype, args.min_run_time)
        membw_res = benchmark_mem_bandwidth(device, args.mem_bandwidth_size, dtype, args.min_run_time)

        # add for compare
        compare_inputs.extend([
            matmul_res[0],
            conv_res[0],
            elem_res[0],
            train_res[0],
            membw_res[0],
        ])

        # print per-dtype summary
        print("\nRaw results (median sec) + peak mem:")
        for res, peak_alloc, peak_reserved in [matmul_res, conv_res, elem_res, train_res]:
            print(
                f"{res.label:12} {res.sub_label:25} {res.median:.6f}s "
                f"[{res.description}] peak_alloc={peak_alloc/1024**2:.2f}MB "
                f"peak_reserved={peak_reserved/1024**2:.2f}MB"
            )
        res, peak_alloc, peak_reserved, bytes_moved = membw_res
        print(
            f"{res.label:12} {res.sub_label:25} {res.median:.6f}s "
            f"[{res.description}] peak_alloc={peak_alloc/1024**2:.2f}MB "
            f"peak_reserved={peak_reserved/1024**2:.2f}MB "
            f"bytes_moved={bytes_moved}"
        )

        if not args.disable_torchbench:
            for tb in torchbench_results:
                res, peak_alloc, peak_reserved, model_name, mode, _dtype = tb
                print(
                    f"{res.label:12} {res.sub_label:25} {res.median:.6f}s "
                    f"[TorchBench {model_name} {mode}] peak_alloc={peak_alloc/1024**2:.2f}MB "
                    f"peak_reserved={peak_reserved/1024**2:.2f}MB"
                )

        # store with dtype for CSV
        all_results.extend([
            (matmul_res[0], matmul_res[1], matmul_res[2], dtype_name),
            (conv_res[0], conv_res[1], conv_res[2], dtype_name),
            (elem_res[0], elem_res[1], elem_res[2], dtype_name),
            (train_res[0], train_res[1], train_res[2], dtype_name),
        ])
        all_membw_results.append(
            (membw_res[0], membw_res[1], membw_res[2], membw_res[3], dtype_name)
        )
        all_torchbench_results.extend(torchbench_results)

    # compare all results together
    compare = benchmark.Compare(compare_inputs)
    compare.print()

    # finally dump to csv
    save_raw_times_to_csv(
        all_results,
        all_membw_results,
        all_torchbench_results,
        args.output,
        torch_version,
    )


if __name__ == "__main__":
    main()
