#!/bin/bash

sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
git submodule update --init --recursive

uv run python src/test_complex.py  --output results/native.csv

sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'

docker run --gpus all -it --entrypoint /usr/bin/bash  -v $(pwd):/workspace  --ulimit memlock=-1 --ulimit stack=67108864  --rm nvcr.io/nvidia/pytorch:25.10-py3  /workspace/docker_run.sh