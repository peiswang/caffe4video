#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <string>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/video_io.hpp"

namespace caffe {

// added destructor definition in video_io.hpp
DatumVideoReader::~DatumVideoReader() {
  if(reader.isOpened()) {
    reader.release();
  }
}

bool DatumVideoReader::ReadVideoToDatum(const string& filename, const int label,
     const int height, const int width, const bool is_color, Datum* datum) {
  reader.open(filename);
  if(!reader.isOpened()) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return false;
  }
  int num_frames = reader.get(CV_CAP_PROP_FRAME_COUNT);
  int _height = reader.get(CV_CAP_PROP_FRAME_HEIGHT);
  int _width = reader.get(CV_CAP_PROP_FRAME_WIDTH);

  bool need_resize = false;
  if (height > 0 && width >0 && (height != _height || width != _width)) {
    need_resize = true;
    _height = height;
    _width = width;
  } 

  int num_channels = (is_color ? 3 : 1);
  datum->set_frames(num_frames);
  datum->set_channels(num_channels);
  datum->set_height(_height);
  datum->set_width(_width);
  datum->set_label(label);
  datum->clear_data();
  datum->clear_float_data();
  string* datum_string = datum->mutable_data();


  for (int frame_id = 0; frame_id < num_frames; ++frame_id) {
    reader>>frame;
    if (!frame.empty()) {
      if (need_resize) {
        cv::resize(frame, img, cv::Size(_width, _height));
      } else {
        img = frame;
      }

      if (is_color) {
        for (int c = 0; c < num_channels; ++c) {
          for (int h = 0; h < _height; ++h) {
            for (int w = 0; w < _width; ++w) {
              datum_string->push_back(
                static_cast<char>(img.at<cv::Vec3b>(h, w)[c]));
            }
          }
        }
      } else {  // Faster than repeatedly testing is_color for each pixel w/i loop
        for (int h = 0; h < _height; ++h) {
          for (int w = 0; w < _width; ++w) {
            datum_string->push_back(
              static_cast<char>(img.at<uchar>(h, w)));
            }
          }
      }

    } else {
      LOG(ERROR) << "empty frame in " << filename;
      reader.release();
      return false;
    }
  }
  reader.release();
  return true;
   
}

inline bool DatumVideoReader::ReadVideoToDatum(const string& filename, const int label,
     const int height, const int width, Datum* datum) {
  return ReadVideoToDatum(filename, label, height, width, true, datum);
}

inline bool DatumVideoReader::ReadVideoToDatum(const string& filename, const int label,
     Datum* datum) {
  return ReadVideoToDatum(filename, label, 0, 0, true, datum);
  
}


}  // namespace caffe
