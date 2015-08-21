#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void TemporalPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  TemporalPoolingParameter pool_param = this->layer_param_.temporal_pooling_param();

  kernel_size_ = pool_param.kernel_size();
  group_ = pool_param.group();
  pad_= pool_param.pad();
  stride_ = pool_param.stride();
  channels_ = bottom[0]->channels();

  vl_ = channels_ / group_;

  CHECK_EQ(channels_ % group_, 0);
  CHECK_GT(kernel_size_, 0) << "Filter dimensions cannot be zero.";

  pooled_length_ = static_cast<int>(ceil(static_cast<float>(group_ + 2 * pad_ - kernel_size_) / stride_)) + 1;

  if (pad_ != 0) {
    CHECK(this->layer_param_.temporal_pooling_param().pool()
        == TemporalPoolingParameter_PoolMethod_AVE
        || this->layer_param_.temporal_pooling_param().pool()
        == TemporalPoolingParameter_PoolMethod_MAX)
        << "Padding implemented only for average and max pooling.";
    CHECK_LT(pad_, kernel_size_);
  }
}

template <typename Dtype>
void TemporalPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  K_ = vl_ * width_ * height_;

  if (pad_) {
    if ((pooled_length_ - 1) * stride_ >= group_ + pad_) {
      --pooled_length_;
    }
    CHECK_LT((pooled_length_- 1) * stride_, group_ + pad_);
  }

  (*top)[0]->Reshape(bottom[0]->num(), pooled_length_ * vl_, height_, width_);

  //if (top->size() > 1) {
  //  (*top)[1]->ReshapeLike(*(*top)[0]);
  //}
  // If max pooling, we will initialize the vector index part.
  if (this->layer_param_.temporal_pooling_param().pool() ==
      TemporalPoolingParameter_PoolMethod_MAX) {
    max_idx_.Reshape(bottom[0]->num(), pooled_length_ * vl_, height_, width_);
  }
  // If stochastic pooling, we will initialize the random index part.
  //if (this->layer_param_.temporal_pooling_param().pool() ==
  //    TemporalPoolingParameter_PoolMethod_STOCHASTIC) {
  //  rand_idx_.Reshape(bottom[0]->num(), pooled_length_ * vl_, height_, width_);
  //}
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void TemporalPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const int top_count = (*top)[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  //const bool use_top_mask = top->size() > 1;
  int* mask = NULL;  // suppress warnings about uninitalized variables
  //Dtype* top_mask = NULL;
  int offset = vl_ * width_ * height_;
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (this->layer_param_.temporal_pooling_param().pool()) {
  case TemporalPoolingParameter_PoolMethod_MAX:
    // Initialize
    //if (use_top_mask) {
    //  top_mask = (*top)[1]->mutable_cpu_data();
    //  caffe_set(top_count, Dtype(-1), top_mask);
    //} else {
    mask = max_idx_.mutable_cpu_data();
    caffe_set(top_count, -1, mask);
    //}
    caffe_set(top_count, Dtype(-FLT_MAX), top_data);
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int g = 0; g < pooled_length_; g++) {
        int offset_ng = (*top)[0]->offset(n) + offset * g;
        if (g * stride_ < pad_) {
          int valid_count = kernel_size_ - pad_ + g * stride_;  // last #valid_count input inside the kernel_size
          for (int index = 0; index < valid_count; index ++) {
            caffe_cpu_vimax(offset, top_data + offset_ng, mask + offset_ng, 
                            bottom_data + bottom[0]->offset(n) + index * offset, 
                            index);
          }
        //} else if (g * stride_ + kernel_size- > pad_ + group_) {
        //  int valid_count = pad_ + group_ - g * stride_;  // first #valid_count input inside the kernel_size
        //  for (int index = 0; index < valid_count; index++) {
        //    caffe_cpu_vimax(offset, top_data + offset_ng, mask + offset_ng, 
        //                    bottom_data + bottom[0]->offset(n) + (g * stride_ - pad_ + index) * offset, 
        //                    index);
        //  }
        } else {
          int valid_count = min(kernel_size_, pad_ + group_ - g * stride_);  
          for (int index = 0; index < valid_count; index++) {
            caffe_cpu_vimax(offset, top_data + offset_ng, mask + offset_ng, 
                            bottom_data + bottom[0]->offset(n) + (g * stride_ - pad_ + index) * offset, 
                            index);
          }
        }
      }
    }
    break;
  case TemporalPoolingParameter_PoolMethod_AVE:
    //for (int i = 0; i < top_count; ++i) {
    //  top_data[i] = 0;
    //}
    //// The main loop
    //for (int n = 0; n < bottom[0]->num(); ++n) {
    //  for (int c = 0; c < channels_; ++c) {
    //    for (int ph = 0; ph < pooled_height_; ++ph) {
    //      for (int pw = 0; pw < pooled_width_; ++pw) {
    //        int hstart = ph * stride_h_ - pad_h_;
    //        int wstart = pw * stride_w_ - pad_w_;
    //        int hend = min(hstart + kernel_h_, height_ + pad_h_);
    //        int wend = min(wstart + kernel_w_, width_ + pad_w_);
    //        int pool_size = (hend - hstart) * (wend - wstart);
    //        hstart = max(hstart, 0);
    //        wstart = max(wstart, 0);
    //        hend = min(hend, height_);
    //        wend = min(wend, width_);
    //        for (int h = hstart; h < hend; ++h) {
    //          for (int w = wstart; w < wend; ++w) {
    //            top_data[ph * pooled_width_ + pw] +=
    //                bottom_data[h * width_ + w];
    //          }
    //        }
    //        top_data[ph * pooled_width_ + pw] /= pool_size;
    //      }
    //    }
    //    // compute offset
    //    bottom_data += bottom[0]->offset(0, 1);
    //    top_data += (*top)[0]->offset(0, 1);
    //  }
    //}
    NOT_IMPLEMENTED;
    break;
  case TemporalPoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

template <typename Dtype>
void TemporalPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set((*bottom)[0]->count(), Dtype(0), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  //const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;  // suppress warnings about uninitialized variables
  int offset = vl_ * width_ * height_;
  //const Dtype* top_mask = NULL;
  switch (this->layer_param_.temporal_pooling_param().pool()) {
  case TemporalPoolingParameter_PoolMethod_MAX:
    // The main loop
    //if (use_top_mask) {
    //  top_mask = top[1]->cpu_data();
    //} else {
    mask = max_idx_.cpu_data();
    //}
    for (int n = 0; n < (*bottom)[0]->num(); ++n) {
      for (int g = 0; g < pooled_length_; g++) {
        int offset_ng = top[0]->offset(n) + offset * g;
        if (g * stride_ < pad_) {
          caffe_cpu_backfill(offset, top_diff + offset_ng, mask + offset_ng, 
                            bottom_diff + (*bottom)[0]->offset(n), true);
        } else {
          caffe_cpu_backfill(offset, top_diff + offset_ng, mask + offset_ng, 
                            bottom_diff + (*bottom)[0]->offset(n) + (g * stride_ - pad_) * offset, true);
        }
      }
    }
    break;
  case TemporalPoolingParameter_PoolMethod_AVE:
    // The main loop
    //for (int n = 0; n < top[0]->num(); ++n) {
    //  for (int c = 0; c < channels_; ++c) {
    //    for (int ph = 0; ph < pooled_height_; ++ph) {
    //      for (int pw = 0; pw < pooled_width_; ++pw) {
    //        int hstart = ph * stride_h_ - pad_h_;
    //        int wstart = pw * stride_w_ - pad_w_;
    //        int hend = min(hstart + kernel_h_, height_ + pad_h_);
    //        int wend = min(wstart + kernel_w_, width_ + pad_w_);
    //        int pool_size = (hend - hstart) * (wend - wstart);
    //        hstart = max(hstart, 0);
    //        wstart = max(wstart, 0);
    //        hend = min(hend, height_);
    //        wend = min(wend, width_);
    //        for (int h = hstart; h < hend; ++h) {
    //          for (int w = wstart; w < wend; ++w) {
    //            bottom_diff[h * width_ + w] +=
    //              top_diff[ph * pooled_width_ + pw] / pool_size;
    //          }
    //        }
    //      }
    //    }
    //    // offset
    //    bottom_diff += (*bottom)[0]->offset(0, 1);
    //    top_diff += top[0]->offset(0, 1);
    //  }
    //}
    NOT_IMPLEMENTED;
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}


#ifdef CPU_ONLY
STUB_GPU(TemporalPoolingLayer);
#endif

INSTANTIATE_CLASS(TemporalPoolingLayer);


}  // namespace caffe
