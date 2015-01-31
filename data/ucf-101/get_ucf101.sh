#!/usr/bin/env sh
# This scripts downloads the UCF-101(binary version) data and unzips it.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd $DIR

echo "Downloading ucf-101 dataset..."

wget --no-check-certificate http://crcv.ucf.edu/data/UCF101/UCF101.rar

echo "Downloading train/test splits for action recognition on ucf-101 dataset..."

wget --no-check-certificate http://crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip

echo "Unzipping..."

rar x UCF101.rar
unzip UCF101TrainTestSplits-RecognitionTask.zip

echo "Done."
