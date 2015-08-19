#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include <algorithm>

namespace caffe {

template <typename Dtype>
void TemporalConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // Configure the kernel size, padding, stride, and inputs.
  // 
  TemporalConvolutionParameter temporal_convolution_param = this->layer_param_.temporal_convolution_param();

  group_ = temporal_convolution_param.group();
  num_output_ = temporal_convolution_param.num_output();
  stride_ = temporal_convolution_param.stride();
  kernel_size_ = temporal_convolution_param.kernel_size();
  pad_ = temporal_convolution_param.pad();

  CHECK_GT(stride_, 0) << "stride must greater than zero";

  // Configure output channels and groups.
  channels_ = bottom[0]->channels();
  CHECK_EQ(channels_ % group_, 0);

  group_out_ = (group_ + 2 * pad_ - kernel_size_) / stride_ + 1; // number of output vectors
  vl_ = channels_ / group_;         // length of output vector

  bias_term_ = temporal_convolution_param.bias_term();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
        this->blobs_.resize(2);
    } else {
        this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // #input x assemble_size x kernel height x kernel width
    this->blobs_[0].reset(new ParamBlob<Dtype>(
            1, num_output_, kernel_size_, vl_));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
            temporal_convolution_param.weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases:
    //
    if (bias_term_) {
      this->blobs_[1].reset(new ParamBlob<Dtype>(
            1, 1, 1, num_output_));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
            temporal_convolution_param.bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void TemporalConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  num_ = bottom[0]->num();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  // TODO: generalize to handle inputs of different shapes.
  for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
    CHECK_EQ(num_, bottom[bottom_id]->num()) << "Inputs must have same num.";
    CHECK_EQ(channels_, bottom[bottom_id]->channels())
        << "Inputs must have same channels.";
    CHECK_EQ(height_, bottom[bottom_id]->height())
        << "Inputs must have same height.";
    CHECK_EQ(width_, bottom[bottom_id]->width())
        << "Inputs must have same width.";
  }
  // Shape the tops.
  for (int top_id = 0; top_id < top->size(); ++top_id) {
    (*top)[top_id]->Reshape(num_, group_out_ * num_output_ , height_, width_);
  }
  // Prepare the matrix multiplication computation.
  // Each input will be calculated as a single GEMM.
  M_ = num_output_;
  N_ = height_ * width_;
  K_ = kernel_size_ * vl_;
  // 
  // Set up padded_bottom_i_ if pad is needed
  if (pad_ > 0) {
    padded_bottom_i_.Reshape(1, K_, height_, width_);
  }
  // Set up the all ones "bias multiplier" for adding biases by BLAS
  if (bias_term_) {
    bias_multiplier_.Reshape(1, 1, 1, N_);
    caffe_set(N_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void TemporalConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  for (int i = 0; i < bottom.size(); ++i) {

    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = (*top)[i]->mutable_cpu_data();
    const Dtype* weight = this->blobs_[0]->cpu_data();

    int bottom_offset = vl_ * N_;  // number of values in an input region
    int top_offset = num_output_ * N_;  // number of values in an output region / column

    for (int n = 0; n < num_; ++n) {
      for (int g = 0; g < group_out_; ++g) {
        // -      -      -      -      #   #   #   #   #   #   #   #      #     -     -     -     -
        // 0                  pad-1   pad                            pad+group-1  pad+group
        if (g * stride_ < pad_) {
          // pad left
          caffe_set((pad_ - g * stride_) * bottom_offset, Dtype(0), padded_bottom_i_.mutable_cpu_data());
          caffe_copy((kernel_size_ - pad_ + g * stride_) * vl_ * N_, 
                          bottom_data + bottom[i]->offset(n), 
                          padded_bottom_i_.mutable_cpu_data() + (pad_ - g * stride_) * bottom_offset);
          caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
              (Dtype)1., weight, padded_bottom_i_.cpu_data(),
              (Dtype)0., top_data + (*top)[i]->offset(n) + top_offset * g);
        } else if (g * stride_ + kernel_size_  > pad_ + group_) {
          // pad right
          caffe_copy((pad_ + group_ - g * stride_) * bottom_offset, 
                     bottom_data + bottom[i]->offset(n) + (g * stride_ - pad_) * bottom_offset,
                     padded_bottom_i_.mutable_cpu_data());
          caffe_set((kernel_size_ + g * stride_ - pad_ - group_) * bottom_offset, Dtype(0), 
                     padded_bottom_i_.mutable_cpu_data() + (pad_ + group_ - g * stride_) * bottom_offset);
          caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
              (Dtype)1., weight, padded_bottom_i_.cpu_data(),
              (Dtype)0., top_data + (*top)[i]->offset(n) + top_offset * g);
        } else {
          // no pad
          caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
              (Dtype)1., weight, bottom_data + bottom[i]->offset(n) + (g * stride_ - pad_) * bottom_offset,
              (Dtype)0., top_data + (*top)[i]->offset(n) + top_offset * g);
        }
        if (bias_term_) {
          caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, 
              N_, 1, (Dtype)1., this->blobs_[1]->cpu_data(),
              bias_multiplier_.cpu_data(),
              (Dtype)1., top_data + (*top)[i]->offset(n) + top_offset * g);
        }
      }
    }

  }
}

template <typename Dtype>
void TemporalConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {

  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->cpu_data();
    weight_diff = this->blobs_[0]->mutable_cpu_diff();
    caffe_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  }
  Dtype* bias_diff = NULL;
  if (bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_cpu_diff();
    caffe_set(this->blobs_[1]->count(), Dtype(0), bias_diff);
  }

  const int top_offset = num_output_ * N_;
  int bottom_offset = vl_ * N_;  // number of values in an input region

  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = (*bottom)[i]->cpu_data();
    Dtype* bottom_diff = (*bottom)[i]->mutable_cpu_diff();
    caffe_set(num_*channels_*height_*width_, Dtype(0), bottom_diff);
    
    if (this->param_propagate_down_[0] || propagate_down[i] || 
                    bias_term_ && this->param_propagate_down_[1]) {
      for (int n = 0; n < num_; ++n) {
        for (int g = 0; g < group_out_; ++g) {

          int offset_ng = top[i]->offset(n) + top_offset * g;
          if (bias_term_ && this->param_propagate_down_[1]) {
            caffe_cpu_gemv<Dtype>(CblasNoTrans, num_output_, N_,
                1., top_diff + offset_ng,
                bias_multiplier_.cpu_data(), 1.,
                bias_diff);
          }
          if (this->param_propagate_down_[0] || propagate_down[i]) {

            // gradient w.r.t. weight. Note that we will accumulate diffs.
            if (this->param_propagate_down_[0]) {
              if (g * stride_ < pad_) {
                // pad left
                caffe_set((pad_ - g * stride_) * bottom_offset, Dtype(0), padded_bottom_i_.mutable_cpu_data());
                caffe_copy((kernel_size_ - pad_ + g * stride_) * bottom_offset, 
                                bottom_data + (*bottom)[i]->offset(n), 
                                padded_bottom_i_.mutable_cpu_data() + (pad_ - g * stride_) * bottom_offset);
                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
                    (Dtype)1., top_diff + offset_ng, padded_bottom_i_.cpu_data(),
                    (Dtype)1., weight_diff);
              } else if (g * stride_ + kernel_size_  > pad_ + group_) {
                // pad right
                caffe_copy((pad_ + group_ - g * stride_) * bottom_offset, 
                           bottom_data + (*bottom)[i]->offset(n) + (g * stride_ - pad_) * bottom_offset,
                           padded_bottom_i_.mutable_cpu_data());
                caffe_set((kernel_size_ + g * stride_ - pad_ - group_) * bottom_offset, Dtype(0), 
                           padded_bottom_i_.mutable_cpu_data() + (pad_ + group_ - g * stride_) * bottom_offset);
                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
                    (Dtype)1., top_diff + offset_ng, padded_bottom_i_.cpu_data(),
                    (Dtype)1., weight_diff);
              } else {
                // no pad
                caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
                    (Dtype)1., top_diff + offset_ng,
                    bottom_data + (*bottom)[i]->offset(n) + (g * stride_ - pad_) * bottom_offset, (Dtype)1.,
                    weight_diff);
              }
            }
            // gradient w.r.t. bottom data, if necessary.
            if (propagate_down[i]) {
              if (weight == NULL) {
                weight = this->blobs_[0]->cpu_data();
              }
              if (g * stride_ < pad_) {
                // pad left
                //caffe_set((pad_ - g * stride_) * bottom_offset, Dtype(0), padded_bottom_i_.mutable_cpu_data());
                caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
                    (Dtype)1., weight,
                    top_diff + offset_ng,
                    (Dtype)0., padded_bottom_i_.mutable_cpu_diff());
                caffe_add((kernel_size_ - pad_ + g * stride_) * bottom_offset, 
                                (*bottom)[i]->cpu_diff() + (*bottom)[i]->offset(n), 
                                padded_bottom_i_.cpu_diff() + (pad_ - g * stride_) * bottom_offset,
                                bottom_diff + (*bottom)[i]->offset(n));
              } else if (g * stride_ + kernel_size_  > pad_ + group_) {
                // pad right
                //caffe_set((kernel_size_ + g * stride_ - pad_ - group_) * bottom_offset, Dtype(0), 
                //           padded_bottom_i_.mutable_cpu_data() + (pad_ + group_ - g * stride_) * bottom_offset);
                caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
                    (Dtype)1., weight,
                    top_diff + offset_ng,
                    (Dtype)0., padded_bottom_i_.mutable_cpu_diff());
                caffe_add((pad_ + group_ - g * stride_) * bottom_offset, 
                           (*bottom)[i]->cpu_diff() + (*bottom)[i]->offset(n) + (g * stride_ - pad_) * bottom_offset,
                           padded_bottom_i_.cpu_diff(),
                           bottom_diff + (*bottom)[i]->offset(n) + (g * stride_ - pad_) * bottom_offset);
              } else {
                //caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
                //    (Dtype)1., top_diff + offset_ng,
                //    bottom_data + (*bottom)[i]->offset(n) + (g * stride_ - pad_) * bottom_offset, (Dtype)1.,
                //    weight_diff);

                // no pad
                caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
                    (Dtype)1., weight,
                    top_diff + offset_ng,
                    (Dtype)1., bottom_diff + (*bottom)[i]->offset(n) + (g * stride_ - pad_) * bottom_offset);
              }
            }
          }
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(TemporalConvolutionLayer);
#endif

INSTANTIATE_CLASS(TemporalConvolutionLayer);

}  // namespace caffe
