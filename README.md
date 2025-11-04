# Perf check for DGX Spark


## Run native

```
uv sync
uv run src/test.py

```

## Run in docker

```
docker run --gpus all -it --entrypoint /usr/bin/bash  -v $(pwd):/workspace  --ulimit memlock=-1 --ulimit stack=67108864  --rm --entrypoint /usr/bin/python nvcr.io/nvidia/pytorch:25.10-p
y3  src/test.py
```