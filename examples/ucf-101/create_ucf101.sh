#!/usr/bin/env sh
# Create the ucf101 lmdb inputs
# N.B. set the path to the ucf101 train + val data dirs

EXAMPLE=examples/ucf-101
DATA=data/ucf-101
TOOLS=build/examples/ucf-101

TRAIN_DATA_ROOT=data/ucf-101/UCF101/
TEST_DATA_ROOT=data/ucf-101/UCF101/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=320
  RESIZE_WIDTH=240
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_ucf101.sh to the path" \
       "where the ucf101 training data is stored."
  exit 1
fi

if [ ! -d "$TEST_DATA_ROOT" ]; then
  echo "Error: TEST_DATA_ROOT is not a path to a directory: $TEST_DATA_ROOT"
  echo "Set the TEST_DATA_ROOT variable in create_ucf101.sh to the path" \
       "where the ucf101 test data is stored."
  exit 1
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_ucf101.bin \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT \
    $DATA/ucfTrainTestlist/trainlist01_.txt \
    $EXAMPLE/ucf101_train_lmdb

echo "Creating test lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_ucf101.bin \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TEST_DATA_ROOT \
    $DATA/ucfTrainTestlist/testlist01_.txt \
    $EXAMPLE/ucf101_test_lmdb

echo "Done."
