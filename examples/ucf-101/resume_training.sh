#!/usr/bin/env sh

./build/tools/caffe train \
    --solver=examples/ucf-101/solver.prototxt \
    --snapshot=examples/ucf-101/caffenet_train_10000.solverstate
