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

    int video_crop_size_h = param_.get_video_crop_size_h();
    int video_crop_size_w = param_.get_video_crop_size_w();
    int crop_frames = param_.get_video_crop_size_t();
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

    if (mirror && crop_size == 0 && (video_crop_size_w ==0 || video_crop_size_h == 0)) {
      LOG(FATAL) << "Current implementation requires mirror and crop_size(or video_crop_size) "
                 << "to be set at the same time.";
    }

    if (crop_width && crop_height) {
      CHECK(data.size()) << "Image cropping only support uint8 data";
      int h_off, w_off, f_off;
      // We only do random crop when we do training.
      if (phase_ == Caffe::TRAIN) {
        h_off = Rand() % (height - crop_height);
        w_off = Rand() % (width - crop_width);
        f_off = Rand() % (frames - crop_frames);
      } else {
        h_off = (height - crop_size) / 2;
        w_off = (width - crop_size) / 2;
        f_off = (frames - crop_frames) / 2;
      }
      if (mirror && Rand() % 2) {
        // Copy mirrored version
        for (int f = 0; f < crop_frames; ++f) {
          for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < crop_size; ++h) {
              for (int w = 0; w < crop_size; ++w) {
                int data_index = (((f_off + f) * channels + c) * height + h + h_off) * width + w + w_off;
                int mean_index = (c * height + h + h_off) * width + w + w_off;
                int top_index = (((batch_item_id * frames + f)* channels + c) * crop_height + h)
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
          for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < crop_size; ++h) {
              for (int w = 0; w < crop_size; ++w) {
                int data_index = (((f_off + f) * channels + c) * height + h + h_off) * width + w + w_off;
                int mean_index = (c * height + h + h_off) * width + w + w_off;
                int top_index = (((batch_item_id * frames + f)* channels + c) * crop_height + h)
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
      if (data.size()) {
        for (int f = 0; f < crop_frames; ++f) {
          for (int j = 0; j < size; ++j) {
            Dtype datum_element =
                static_cast<Dtype>(static_cast<uint8_t>(data[f * size + j]));
            transformed_data[f * size + j + batch_item_id * frames * size] =
                (datum_element - mean[j]) * scale;
          }
        }
      } else {
        for (int f = 0; f < crop_frames; ++f) {
          for (int j = 0; j < size; ++j) {
            transformed_data[f * size + j + batch_item_id * frames * size] =
                (datum.float_data(f * size + j) - mean[j]) * scale;
          }
        }
      }
    }
  
  }
}

template <typename Dtype>
void DataTransformer<Dtype>::InitRand() {
  const bool needs_rand = (phase_ == Caffe::TRAIN) &&
      (param_.mirror() || param_.crop_size());
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
