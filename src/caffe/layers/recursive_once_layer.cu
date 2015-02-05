#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include <algorithm>

namespace caffe {

template <typename Dtype>
void RecursiveOnceLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  for (int i = 0; i < bottom.size(); ++i) {

    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = (*top)[i]->mutable_gpu_data();
    int* mask = NULL;
    mask = max_idx_.mutable_gpu_data();
    Dtype* weight_buf_data = weight_buffer_.mutable_gpu_data(); // reshaped weight matrix
    Dtype* out_data = tmp_buffer_.mutable_gpu_data();       // tmp output
    const Dtype* weight = this->blobs_[0]->gpu_data();

    //int weight_offset = M_ * K_;  // number of filter parameters in a group
    int across_offset = vl_ * stride_ * N_;  // number of values in an input region
    int top_offset = vl_ * N_;  // number of values in an output region / column
    int src_weight_offset = vl_ * vl_ * num_uv_;  // number of values in one set of src weight 
    
    // reshape weight matrix  
    //   -- dst weight matrix [assemble_size*vl , across*vl]
    //   -- src weight matrix [assemble_size , num_uv, vl, vl]
    caffe_set(M_ * K_, Dtype(0.0), weight_buf_data);
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
        caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
            (Dtype)1., weight_buf_data , bottom_data + bottom[i]->offset(n) + across_offset * g,
            (Dtype)0., out_data);
        if (bias_term_) {
          caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, 
              N_, 1, (Dtype)1., this->blobs_[1]->gpu_data(),
              bias_multiplier_.gpu_data(),
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
          // uncomment to test mean-out for gradient check (?)
          //for(int sdf=0;sdf<top_offset;sdf++)
          //        top_data_ng[sdf] /= assemble_size_;
        }
      }
    }

  }
}

template <typename Dtype>
void RecursiveOnceLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {

  Dtype* tmp_diff = tmp_buffer_.mutable_gpu_diff();
  const int* mask = max_idx_.gpu_data();
  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->gpu_data();
    weight_diff = this->blobs_[0]->mutable_gpu_diff();
    caffe_set(this->blobs_[0]->count(), Dtype(0), weight_diff);
  }
  Dtype* bias_diff = NULL;
  if (bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
    caffe_set(this->blobs_[1]->count(), Dtype(0), bias_diff);
  }

  const Dtype* weight_buf_data = weight_buffer_.gpu_data(); // reshaped weight matrix
  Dtype* weight_buf_diff = weight_buffer_.mutable_gpu_diff(); // reshaped weight matrix diff
  caffe_set(M_ * K_, Dtype(0), weight_buf_diff);

  const int top_offset = vl_ * N_;
  int across_offset = vl_ * stride_ * N_;  // number of values in an input region
  int src_weight_offset = vl_ * vl_ * num_uv_;  // number of values in one set of src weight 

  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    const Dtype* bottom_data = (*bottom)[i]->gpu_data();
    Dtype* bottom_diff = (*bottom)[i]->mutable_gpu_diff();
    caffe_set(num_*channels_*height_*width_, Dtype(0), bottom_diff);
    
    if (this->param_propagate_down_[0] || propagate_down[i] || 
                    bias_term_ && this->param_propagate_down_[1]) {
      for (int n = 0; n < num_; ++n) {
        for (int g = 0; g < group_out_; ++g) {
          // top to tmp
          caffe_set(M_*N_, Dtype(0.0), tmp_diff);
          int offset_ng = top[0]->offset(n) + top_offset * g;
          caffe_gpu_backfill(top_offset, top_diff + offset_ng,
                             mask + offset_ng, tmp_diff);
          // Bias gradient, if necessary.
          if (bias_term_ && this->param_propagate_down_[1]) {
            caffe_gpu_gemv<Dtype>(CblasNoTrans, vl_ * assemble_size_, N_,
                1., tmp_diff,
                bias_multiplier_.gpu_data(), 1.,
                bias_diff);
          }
          if (this->param_propagate_down_[0] || propagate_down[i]) {

            // gradient w.r.t. weight. Note that we will accumulate diffs.
            if (this->param_propagate_down_[0]) {
              caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, K_, N_,
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
                weight = this->blobs_[0]->gpu_data();
              }
              caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, K_, N_, M_,
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


INSTANTIATE_CLASS(RecursiveOnceLayer);

}  // namespace caffe
