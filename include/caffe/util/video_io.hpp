#ifndef CAFFE_UTIL_VIDEO_IO_H_
#define CAFFE_UTIL_VIDEO_IO_H_

#include <unistd.h>
#include <string>

#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

class DatumVideoReader{
public:
  bool ReadVideoToDatum(const string& filename, const int label,
       const int height, const int width, const bool is_color, Datum* datum);
  inline bool ReadVideoToDatum(const string& filename, const int label,
       const int height, const int width, Datum* datum);
  inline bool ReadVideoToDatum(const string& filename, const int label,
       Datum* datum);
private:
  cv::Mat frame;
  cv::Mat img;  // after resize
  cv::VideoCapture reader;
}


}  // namespace caffe

#endif   // CAFFE_UTIL_VIDEO_IO_H_
