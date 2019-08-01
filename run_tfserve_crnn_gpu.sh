#!/bin/bash
set -eux

MODELS_ROOT=${MODELS_ROOT:-"/tmp/models"}

docker run \
    --runtime=nvidia \
    --name tfserve_crnn_gpu \
    --publish 18521:8501 \
    --publish 18520:8500 \
    --mount type=bind,source="$MODELS_ROOT/crnn",target=/models/crnn \
    --env MODEL_NAME=crnn \
    --env LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64 \
    --env CUDA_VISIBLE_DEVICES=2 \
    --env TF_CPP_MIN_VLOG_LEVEL=0 \
    --tty \
    eldon/tensorflow-serving-gpu
