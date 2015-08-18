#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include <algorithm>

namespace caffe {

template <typename Dtype>
void TemporalConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  for (int i = 0; i < bottom.size(); ++i) {

    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = (*top)[i]->mutable_gpu_data();
    const Dtype* weight = this->blobs_[0]->gpu_data();

    int bottom_offset = vl_ * stride_ * N_;  // number of values in an input region
    int top_offset = num_output_ * N_;  // number of values in an output region / column

    for (int n = 0; n < num_; ++n) {
      for (int g = 0; g < group_out_; ++g) {
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
            (Dtype)1., weight, bottom_data + bottom[i]->offset(n) + bottom_offset * g,
            (Dtype)0., top_data + (*top)[i]->offset(n) + top_offset * g);
        if (bias_term_) {
          caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, 
              N_, 1, (Dtype)1., this->blobs_[1]->gpu_data(),
              bias_multiplier_.gpu_data(),
              (Dtype)1., top_data + (*top)[i]->offset(n) + top_offset * g);
        }
      }
    }

  }
}

template <typename Dtype>
void TemporalConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {

  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->gpu_data();
    weight_diff = this->blobs_[0]->mutable_gpu_diff();
    caffe_gpu_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  }
  Dtype* bias_diff = NULL;
  if (bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
    caffe_gpu_set(this->blobs_[1]->count(), Dtype(0), bias_diff);
  }

  const int top_offset = num_output_ * N_;
  int bottom_offset = vl_ * stride_ * N_;  // number of values in an input region

  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    const Dtype* bottom_data = (*bottom)[i]->gpu_data();
    Dtype* bottom_diff = (*bottom)[i]->mutable_gpu_diff();
    caffe_gpu_set(num_*channels_*height_*width_, Dtype(0), bottom_diff);
    
    if (this->param_propagate_down_[0] || propagate_down[i] || 
                    bias_term_ && this->param_propagate_down_[1]) {
      for (int n = 0; n < num_; ++n) {
        for (int g = 0; g < group_out_; ++g) {

          int offset_ng = top[0]->offset(n) + top_offset * g;
          if (bias_term_ && this->param_propagate_down_[1]) {
            caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, N_,
                1., top_diff + offset_ng,
                bias_multiplier_.gpu_data(), 1.,
                bias_diff);
          }
          if (this->param_propagate_down_[0] || propagate_down[i]) {

            // gradient w.r.t. weight. Note that we will accumulate diffs.
            if (this->param_propagate_down_[0]) {
              caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
                  (Dtype)1., top_diff + offset_ng,
                  bottom_data + (*bottom)[i]->offset(n) + g * bottom_offset, (Dtype)1.,
                  weight_diff);
            }
            // gradient w.r.t. bottom data, if necessary.
            if (propagate_down[i]) {
              if (weight == NULL) {
                weight = this->blobs_[0]->gpu_data();
              }
              caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
                  (Dtype)1., weight,
                  top_diff + offset_ng,
                  (Dtype)1., bottom_diff + (*bottom)[i]->offset(n) + g * bottom_offset);
            }
          }
        }
      }
    }
  }
}


INSTANTIATE_CLASS(TemporalConvolutionLayer);

}  // namespace caffe
