#!/usr/bin/env sh
# Compute the mean image from the ucf101 training leveldb
# N.B. this is available in data/ucf-101

./build/examples/ucf-101/compute_video_mean examples/ucf-101/ucf101_train_lmdb \
  examples/ucf-101/ucf101_mean.binaryproto

echo "Done."
