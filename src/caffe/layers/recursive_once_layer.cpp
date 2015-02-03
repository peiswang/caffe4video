#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include <algorithm>

namespace caffe {

template <typename Dtype>
void RecursiveOnceLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  // Configure the kernel size, padding, stride, and inputs.
  // 
  RecursiveOnceParameter recursive_once_param = this->layer_param_.recursive_once_param();

  group_ = recursive_once_param.group();
  num_uv_ = recursive_once_param.num_uv();
  assemble_size_ = recursive_once_param.assemble_size();
  stride_ = recursive_once_param.stride();
  correspond_ = recursive_once_param.correspond();

  CHECK_GT(num_uv_, 0) << "num_uv must greater than zero";
  CHECK_EQ(recursive_once_param.relative_position_size(), num_uv_) 
          << "relative_position size must be equal to num_uv";
  CHECK_GT(stride_, 0) << "stride must greater than zero";

  if (assemble_size_ > 1)
    multi_weights_ = true;
  else
    multi_weights_ = false;

  int first_pos = recursive_once_param.relative_position(0);
  relative_position_.clear();
  relative_position_.push_back(0);
  for (int i=1; i < num_uv_; ++i) {
    int now = recursive_once_param.relative_position(i) - first_pos;
    CHECK_GT(now, relative_position_[i-1]);
    relative_position_.push_back(now);
  }

  across_ = relative_position_.back() + 1;
  CHECK_GE(group_, across_) << "group_ size cannot be smaller than 'across'";

  // Configure output channels and groups.
  channels_ = bottom[0]->channels();
  CHECK_EQ(channels_ % group_, 0);

  group_out_ = (group_ - across_) / stride_ + 1; // number of output vectors
  vl_ = channels_ / group_;         // length of output vector

  bias_term_ = recursive_once_param.bias_term();
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
    this->blobs_[0].reset(new Blob<Dtype>(
            assemble_size_, num_uv_, vl_, vl_));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
            recursive_once_param.weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases:
    //
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(
            1, 1, assemble_size_, vl_));
            //assemble_size_, num_uv_, 1, vector_length));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.convolution_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }
  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void RecursiveOnceLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  num_ = bottom[0]->num();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  CHECK_EQ(bottom[0]->channels(), channels_) << "Input size incompatible with"
    " convolution kernel.";
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
    (*top)[top_id]->Reshape(num_, vl_ * group_out_, height_, width_);
  }
  //if(multi_weights_)
    max_idx_.Reshape(num_, vl_ * group_out_, height_, width_);
  // Prepare the matrix multiplication computation.
  // Each input will be convolved as a single GEMM.
  M_ = assemble_size_ * vl_;
  K_ = across_ * vl_;
  N_ = height_ * width_;
  // 
  //
  weight_buffer_.Reshape(1, 1, M_, K_);
  tmp_buffer_.Reshape(1, M_, height_, width_);
  //int tmp_size = M_ * N_ >= group_out_ ? M_ * N_ : group_out_;
  //tmp_buffer_.Reshape(1, 1, 1, tmp_size);

  // Set up the all ones "bias multiplier" for adding biases by BLAS
  if (bias_term_) {
    bias_multiplier_.Reshape(1, 1, 1, N_);
    caffe_set(N_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void RecursiveOnceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  for (int i = 0; i < bottom.size(); ++i) {

    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = (*top)[i]->mutable_cpu_data();
    int* mask = NULL;
    mask = max_idx_.mutable_cpu_data();
    Dtype* weight_buf_data = weight_buffer_.mutable_cpu_data(); // reshaped weight matrix
    Dtype* out_data = tmp_buffer_.mutable_cpu_data();       // tmp output
    const Dtype* weight = this->blobs_[0]->cpu_data();

    //int weight_offset = M_ * K_;  // number of filter parameters in a group
    int across_offset = vl_ * stride_ * N_;  // number of values in an input region
    int top_offset = vl_ * N_;  // number of values in an output region / column
    int src_weight_offset = vl_ * vl_ * num_uv_;  // number of values in one set of src weight 
    
    // reshape weight matrix  
    //   -- dst weight matrix [assemble_size*vl , across*vl]
    //   -- src weight matrix [assemble_size , num_uv, vl, vl]
    for(int as = 0; as < assemble_size_; ++as) {
      for(int h = 0; h < vl_; ++h) {
        for (int wc = 0; wc < num_uv_; ++wc) {
          caffe_copy(vl_, weight+src_weight_offset * as + vl_*vl_*wc + vl_*h,
                     weight_buf_data+(as*vl_+h)*K_+vl_*relative_position_[wc]);
        }
      }
    }

    for (int n = 0; n < num_; ++n) {
      for (int g = 0; g < group_out_; ++g) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
            (Dtype)1., weight_buf_data , bottom_data + bottom[i]->offset(n) + across_offset * g,
            (Dtype)0., out_data);
        if (bias_term_) {
          caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, 
              N_, 1, (Dtype)1., this->blobs_[1]->cpu_data(),
              bias_multiplier_.cpu_data(),
              (Dtype)1., out_data);
        }
        // max-out to top
        Dtype * top_data_ng = top_data + (*top)[i]->offset(n) + top_offset * g;
        caffe_copy(top_offset, out_data, top_data_ng);
        int* mask_ng = mask + max_idx_.offset(n) + top_offset * g;
        caffe_set(top_offset, 0, mask_ng);
        if (multi_weights_) {
          for (int nid = 1; nid < assemble_size_; ++nid) {
            caffe_vimax(top_offset, top_data_ng, mask_ng, out_data + top_offset * nid, nid);
          }
        }
      }
    }

  }
}

template <typename Dtype>
void RecursiveOnceLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {

  Dtype* tmp_diff = tmp_buffer_.mutable_cpu_diff();
  const int* mask = max_idx_.cpu_data();
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

  Dtype* weight_buf_data = weight_buffer_.mutable_cpu_data(); // reshaped weight matrix
  Dtype* weight_buf_diff = weight_buffer_.mutable_cpu_diff(); // reshaped weight matrix diff
  caffe_set(M_ * K_, Dtype(0), weight_buf_diff);

  const int top_offset = vl_ * N_;
  int across_offset = vl_ * stride_ * N_;  // number of values in an input region
  int src_weight_offset = vl_ * vl_ * num_uv_;  // number of values in one set of src weight 

  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = (*bottom)[i]->cpu_data();
    Dtype* bottom_diff = (*bottom)[i]->mutable_cpu_diff();
    caffe_set(num_*channels_*height_*width_, Dtype(0), bottom_diff);
    
    if (this->param_propagate_down_[0] || propagate_down[i] || 
                    bias_term_ && this->param_propagate_down_[1]) {
      for (int n = 0; n < num_; ++n) {
        for (int g = 0; g < group_out_; ++g) {
          // top to tmp
          int offset_ng = top[0]->offset(n) + top_offset * g;
          caffe_cpu_backfill(top_offset, top_diff + offset_ng,
                             mask + offset_ng, tmp_diff);
          // Bias gradient, if necessary.
          if (bias_term_ && this->param_propagate_down_[1]) {
            caffe_cpu_gemv<Dtype>(CblasNoTrans, vl_ * assemble_size_, N_,
                1., tmp_diff,
                bias_multiplier_.cpu_data(), 1.,
                bias_diff);
          }
          if (this->param_propagate_down_[0] || propagate_down[i]) {

            // gradient w.r.t. weight. Note that we will accumulate diffs.
            if (this->param_propagate_down_[0]) {
              caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
                  (Dtype)1., tmp_diff,
                  bottom_data + (*bottom)[i]->offset(n) + g * across_offset, (Dtype)1.,
                  weight_buf_diff);
            }
            // weight_buf diff back to the weight diff 
            // reshape weight matrix  
            //   -- dst weight matrix [assemble_size*vl , across*vl]
            //   -- src weight matrix [assemble_size , num_uv, vl, vl]
            for(int as = 0; as < assemble_size_; ++as) {
              for(int h = 0; h < vl_; ++h) {
                for (int wc = 0; wc < num_uv_; ++wc) {
                  caffe_copy(vl_,
                             weight_buf_diff+(as*vl_+h)*K_+vl_*relative_position_[wc],
                             weight_diff+src_weight_offset * as + vl_*vl_*wc + vl_*h);
                }
              }
            }
            // gradient w.r.t. bottom data, if necessary.
            if (propagate_down[i]) {
              if (weight == NULL) {
                weight = this->blobs_[0]->cpu_data();
              }
              caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
                  (Dtype)1., weight_buf_data,
                  tmp_diff,
                  (Dtype)1., bottom_diff + (*bottom)[i]->offset(n) + g * across_offset);
            }
          }
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(RecursiveOnceLayer);
#endif

INSTANTIATE_CLASS(RecursiveOnceLayer);

}  // namespace caffe
