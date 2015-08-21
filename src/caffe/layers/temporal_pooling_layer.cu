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
void TemporalPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
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
    mask = max_idx_.mutable_gpu_data();
    caffe_gpu_set(top_count, -1, mask);
    caffe_gpu_set(top_count, Dtype(-FLT_MAX), top_data);
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int g = 0; g < pooled_length_; g++) {
        int offset_ng = (*top)[0]->offset(n) + offset * g;
        if (g * stride_ < pad_) {
          int valid_count = kernel_size_ - pad_ + g * stride_;  // last #valid_count input inside the kernel_size
          for (int index = 0; index < valid_count; index ++) {
            caffe_gpu_vimax(offset, top_data + offset_ng, mask + offset_ng, 
                            bottom_data + bottom[0]->offset(n) + index * offset, 
                            index);
          }
        } else {
          int valid_count = min(kernel_size_, pad_ + group_ - g * stride_);  
          for (int index = 0; index < valid_count; index++) {
            caffe_gpu_vimax(offset, top_data + offset_ng, mask + offset_ng, 
                            bottom_data + bottom[0]->offset(n) + (g * stride_ - pad_ + index) * offset, 
                            index);
          }
        }
      }
    }
    break;
  case TemporalPoolingParameter_PoolMethod_AVE:
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
void TemporalPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_gpu_set((*bottom)[0]->count(), Dtype(0), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  //const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;  // suppress warnings about uninitialized variables
  int offset = vl_ * width_ * height_;
  //const Dtype* top_mask = NULL;
  switch (this->layer_param_.temporal_pooling_param().pool()) {
  case TemporalPoolingParameter_PoolMethod_MAX:
    // The main loop
    //if (use_top_mask) {
    //  top_mask = top[1]->gpu_data();
    //} else {
    mask = max_idx_.gpu_data();
    //}
    for (int n = 0; n < (*bottom)[0]->num(); ++n) {
      for (int g = 0; g < pooled_length_; g++) {
        int offset_ng = top[0]->offset(n) + offset * g;
        if (g * stride_ < pad_) {
          caffe_gpu_backfill(offset, top_diff + offset_ng, mask + offset_ng, 
                            bottom_diff + (*bottom)[0]->offset(n), true);
        } else {
          caffe_gpu_backfill(offset, top_diff + offset_ng, mask + offset_ng, 
                            bottom_diff + (*bottom)[0]->offset(n) + (g * stride_ - pad_) * offset, true);
        }
      }
    }
    break;
  case TemporalPoolingParameter_PoolMethod_AVE:
    NOT_IMPLEMENTED;
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

INSTANTIATE_CLASS(TemporalPoolingLayer);

}  // namespace caffe
