#include <string>

#include "caffe/data_transformer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template<typename Dtype>
void DataTransformer<Dtype>::Transform(const int batch_item_id,
                                       const Datum& datum,
                                       const Dtype* mean,
                                       Dtype* transformed_data) {
  const string& data = datum.data();
  const int frames = datum.frames();
  const int channels = datum.channels();
  const int height = datum.height();
  const int width = datum.width();
  const int size = datum.channels() * datum.height() * datum.width();

  const int crop_size = param_.crop_size();
  const bool mirror = param_.mirror();
  const Dtype scale = param_.scale();

  const bool is_video = param_.is_video();

  if (!is_video) {

    if (mirror && crop_size == 0) {
      LOG(FATAL) << "Current implementation requires mirror and crop_size to be "
                 << "set at the same time.";
    }

    if (crop_size) {
      CHECK(data.size()) << "Image cropping only support uint8 data";
      int h_off, w_off;
      // We only do random crop when we do training.
      if (phase_ == Caffe::TRAIN) {
        h_off = Rand() % (height - crop_size);
        w_off = Rand() % (width - crop_size);
      } else {
        h_off = (height - crop_size) / 2;
        w_off = (width - crop_size) / 2;
      }
      if (mirror && Rand() % 2) {
        // Copy mirrored version
        for (int c = 0; c < channels; ++c) {
          for (int h = 0; h < crop_size; ++h) {
            for (int w = 0; w < crop_size; ++w) {
              int data_index = (c * height + h + h_off) * width + w + w_off;
              int top_index = ((batch_item_id * channels + c) * crop_size + h)
                  * crop_size + (crop_size - 1 - w);
              Dtype datum_element =
                  static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
              transformed_data[top_index] =
                  (datum_element - mean[data_index]) * scale;
            }
          }
        }
      } else {
        // Normal copy
        for (int c = 0; c < channels; ++c) {
          for (int h = 0; h < crop_size; ++h) {
            for (int w = 0; w < crop_size; ++w) {
              int top_index = ((batch_item_id * channels + c) * crop_size + h)
                  * crop_size + w;
              int data_index = (c * height + h + h_off) * width + w + w_off;
              Dtype datum_element =
                  static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
              transformed_data[top_index] =
                  (datum_element - mean[data_index]) * scale;
            }
          }
        }
      }
    } else {
      // we will prefer to use data() first, and then try float_data()
      if (data.size()) {
        for (int j = 0; j < size; ++j) {
          Dtype datum_element =
              static_cast<Dtype>(static_cast<uint8_t>(data[j]));
          transformed_data[j + batch_item_id * size] =
              (datum_element - mean[j]) * scale;
        }
      } else {
        for (int j = 0; j < size; ++j) {
          transformed_data[j + batch_item_id * size] =
              (datum.float_data(j) - mean[j]) * scale;
        }
      }
    }
  } else {

    int video_crop_size_h = param_.video_crop_size_h();
    int video_crop_size_w = param_.video_crop_size_w();
    int crop_frames = param_.video_crop_size_t();

    int video_step_max = param_.video_step_max();

    int crop_width = 0;
    int crop_height = 0;

    if (crop_frames <= 0) {
      LOG(FATAL) << "Video_crop_size_t must be set.";
      return;
    }

    if (video_crop_size_h != 0 && video_crop_size_w !=0) {
      crop_width = video_crop_size_w;
      crop_height = video_crop_size_h;
    } else {
      crop_width = crop_height = crop_size;
    }

    if (mirror && (crop_width <=0 || crop_height <= 0)) {
      LOG(FATAL) << "Current implementation requires mirror and crop_size(or video_crop_size) "
                 << "to be set at the same time.";
    }

    int h_off, w_off, f_off, left_index, right_index;

    // video status
    bool need_pad = false;
    int v_step = frames / crop_frames;
    if(v_step == 0) {
      need_pad =  true;
      v_step = 1;
    } else if(v_step > video_step_max) {
      v_step = video_step_max;
    }
    if(v_step >= 2) {
      v_step = Rand() % v_step + 1;
    }

    if(need_pad) {
      left_index = (crop_frames - frames) / 2;
      right_index = left_index + crop_frames - 1;
      f_off = - left_index;
    } else {
      f_off = Rand() % (frames - crop_frames * v_step + v_step);
    }

    // memset transformed data
    int video_size = crop_frames * channels * crop_height * crop_width;
    caffe_set(video_size, Dtype(0), 
              transformed_data + (batch_item_id * video_size));

    if (crop_width > 0 && crop_height > 0) {
      CHECK(data.size()) << "Image cropping only support uint8 data";

      // We only do random crop when we do training.
      if (phase_ == Caffe::TRAIN) {
        h_off = Rand() % (height - crop_height);
        w_off = Rand() % (width - crop_width);
      } else {
        h_off = (height - crop_height) / 2;
        w_off = (width - crop_width) / 2;
      }
      if (mirror && Rand() % 2) {
        // Copy mirrored version
        for (int f = 0; f < crop_frames; ++f) {
          if(need_pad && (f<left_index || f>right_index)) {
            continue;
          }
          // cf: coresponding f
          int cf = f * v_step;
          for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < crop_height; ++h) {
              for (int w = 0; w < crop_width; ++w) {
                int data_index = (((f_off + cf) * channels + c) * height + h + h_off) * width + w + w_off;
                int mean_index = (c * height + h + h_off) * width + w + w_off;
                int top_index = (((batch_item_id * crop_frames + f)* channels + c) * crop_height + h)
                    * crop_width + (crop_width - 1 - w);
                Dtype datum_element =
                    static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
                transformed_data[top_index] =
                    (datum_element - mean[mean_index]) * scale;
              }
            }
          }
        }
      } else {
        // Normal copy
        for (int f = 0; f < crop_frames; ++f) {
          if(need_pad && (f<left_index || f>right_index)) {
            continue;
          }
          // cf: coresponding f
          int cf = f * v_step;
          for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < crop_height; ++h) {
              for (int w = 0; w < crop_width; ++w) {
                int data_index = (((f_off + cf) * channels + c) * height + h + h_off) * width + w + w_off;
                int mean_index = (c * height + h + h_off) * width + w + w_off;
                int top_index = (((batch_item_id * crop_frames + f)* channels + c) * crop_height + h)
                    * crop_width + w;
                Dtype datum_element =
                    static_cast<Dtype>(static_cast<uint8_t>(data[data_index]));
                transformed_data[top_index] =
                    (datum_element - mean[mean_index]) * scale;
              }
            }
          }
        }
      }
    } else {
      // we will prefer to use data() first, and then try float_data()
      // We only do random crop when we do training.
      if (data.size()) {
        for (int f = 0; f < crop_frames; ++f) {
          if(need_pad && (f<left_index || f>right_index)) {
            continue;
          }
          // cf: coresponding f
          int cf = f * v_step;
          for (int j = 0; j < size; ++j) {
            Dtype datum_element =
                static_cast<Dtype>(static_cast<uint8_t>(data[(f_off + cf) * size + j]));
            transformed_data[(batch_item_id * crop_frames + f) * size + j] =
                (datum_element - mean[j]) * scale;
          }
        }
      } else {
        for (int f = 0; f < crop_frames; ++f) {
          if(need_pad && (f<left_index || f>right_index)) {
            continue;
          }
          // cf: coresponding f
          int cf = f * v_step;
          for (int j = 0; j < size; ++j) {
            transformed_data[(batch_item_id * crop_frames + f) * size + j] =
                (datum.float_data((f_off + cf) * size + j) - mean[j]) * scale;
          }
        }
      }
    }
  
  }
}

template <typename Dtype>
void DataTransformer<Dtype>::InitRand() {
  const bool needs_rand = (phase_ == Caffe::TRAIN) &&
      (param_.mirror() || param_.crop_size() || param_.is_video());
  if (needs_rand) {
    const unsigned int rng_seed = caffe_rng_rand();
    rng_.reset(new Caffe::RNG(rng_seed));
  } else {
    rng_.reset();
  }
}

template <typename Dtype>
unsigned int DataTransformer<Dtype>::Rand() {
  CHECK(rng_);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return (*rng)();
}

INSTANTIATE_CLASS(DataTransformer);

}  // namespace caffe
