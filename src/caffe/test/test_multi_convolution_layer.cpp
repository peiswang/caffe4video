#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

// Reference convolution for checking results:
// accumulate through explicit loops over input, output, and filters.
template <typename Dtype>
void caffe_conv(const Blob<Dtype>* in, MultiConvolutionParameter* conv_param,
    const vector<shared_ptr<Blob<Dtype> > >& weights,
    Blob<Dtype>* out) {
  // Kernel size, stride, and pad
  int kernel_h, kernel_w;
  if (conv_param->has_kernel_size()) {
    kernel_h = kernel_w = conv_param->kernel_size();
  } else {
    kernel_h = conv_param->kernel_h();
    kernel_w = conv_param->kernel_w();
  }
  int pad_h, pad_w;
  if (!conv_param->has_pad_h()) {
    pad_h = pad_w = conv_param->pad();
  } else {
    pad_h = conv_param->pad_h();
    pad_w = conv_param->pad_w();
  }
  int stride_h, stride_w;
  if (!conv_param->has_stride_h()) {
    stride_h = stride_w = conv_param->stride();
  } else {
    stride_h = conv_param->stride_h();
    stride_w = conv_param->stride_w();
  }
  // Groups
  int groups = conv_param->group();
  int o_g = out->channels() / groups;
  int k_g = in->channels() / groups;
  int o_head, k_head;
  // Convolution
  const Dtype* in_data = in->cpu_data();
  const Dtype* weight_data = weights[0]->cpu_data();
  Dtype* out_data = out->mutable_cpu_data();
  for (int n = 0; n < out->num(); n++) {
    for (int g = 0; g < groups; g++) {
      o_head = o_g * g;
      k_head = k_g * g;
      for (int o = 0; o < o_g; o++) {
        for (int k = 0; k < k_g; k++) {
          for (int y = 0; y < out->height(); y++) {
            for (int x = 0; x < out->width(); x++) {
              for (int p = 0; p < kernel_h; p++) {
                for (int q = 0; q < kernel_w; q++) {
                  int in_y = y * stride_h - pad_h + p;
                  int in_x = x * stride_w - pad_w + q;
                  if (in_y >= 0 && in_y < in->height()
                    && in_x >= 0 && in_x < in->width()) {
                    out_data[out->offset(n, o + o_head, y, x)] +=
                        in_data[in->offset(n, k + k_head, in_y, in_x)]
                        * weight_data[weights[0]->offset(o, k, p, q)];
                        //* weight_data[weights[0]->offset(o + o_head, k, p, q)];
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  // Bias
  if (conv_param->bias_term()) {
    const Dtype* bias_data = weights[1]->cpu_data();
    for (int n = 0; n < out->num(); n++) {
      for (int o = 0; o < out->channels(); o++) {
        for (int y = 0; y < out->height(); y++) {
          for (int x = 0; x < out->width(); x++) {
<<<<<<< HEAD
            out_data[out->offset(n, o, y, x)] += bias_data[o % (out->channels()/groups)];
=======
            out_data[out->offset(n, o, y, x)] += bias_data[o % groups];
>>>>>>> upstream/master
          }
        }
      }
    }
  }
}

template void caffe_conv(const Blob<float>* in,
<<<<<<< HEAD
    MultiConvolutionParameter* conv_param,
    const vector<shared_ptr<Blob<float> > >& weights,
    Blob<float>* out);
template void caffe_conv(const Blob<double>* in,
    MultiConvolutionParameter* conv_param,
=======
    ConvolutionParameter* conv_param,
    const vector<shared_ptr<Blob<float> > >& weights,
    Blob<float>* out);
template void caffe_conv(const Blob<double>* in,
    ConvolutionParameter* conv_param,
>>>>>>> upstream/master
    const vector<shared_ptr<Blob<double> > >& weights,
    Blob<double>* out);

template <typename TypeParam>
class MultiConvolutionLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  MultiConvolutionLayerTest()
<<<<<<< HEAD
      : blob_bottom_(new Blob<Dtype>(2, 3*6, 6, 6)), // 3x5
        blob_bottom_2_(new Blob<Dtype>(2, 3*6, 6, 6)),
=======
      : blob_bottom_(new Blob<Dtype>(2, 3*5, 6, 4)), // 3x5
        blob_bottom_2_(new Blob<Dtype>(2, 3*5, 6, 4)),
>>>>>>> upstream/master
        blob_top_(new Blob<Dtype>()),
        blob_top_2_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_value(1.);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->blob_bottom_2_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }

<<<<<<< HEAD
  virtual ~MultiConvolutionLayerTest() {
=======
  virtual ~ConvolutionLayerTest() {
>>>>>>> upstream/master
    delete blob_bottom_;
    delete blob_bottom_2_;
    delete blob_top_;
    delete blob_top_2_;
  }

  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_2_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_2_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(MultiConvolutionLayerTest, TestDtypesAndDevices);

TYPED_TEST(MultiConvolutionLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MultiConvolutionParameter* convolution_param =
      layer_param.mutable_multi_convolution_param();




  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
<<<<<<< HEAD
  convolution_param->set_num_output(4*6);
  convolution_param->set_group(6);
=======
  convolution_param->set_num_output(4*5);
  convolution_param->set_group(5);
>>>>>>> upstream/master

  this->blob_bottom_vec_.push_back(this->blob_bottom_2_);
  this->blob_top_vec_.push_back(this->blob_top_2_);

  shared_ptr<Layer<Dtype> > layer(
      new MultiConvolutionLayer<Dtype>(layer_param));

  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 2);
<<<<<<< HEAD
  EXPECT_EQ(this->blob_top_->channels(), 4*6);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 2);
  EXPECT_EQ(this->blob_top_2_->num(), 2);
  EXPECT_EQ(this->blob_top_2_->channels(), 4*6);
  EXPECT_EQ(this->blob_top_2_->height(), 2);
  EXPECT_EQ(this->blob_top_2_->width(), 2);
=======
  EXPECT_EQ(this->blob_top_->channels(), 4*5);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_2_->num(), 2);
  EXPECT_EQ(this->blob_top_2_->channels(), 4*5);
  EXPECT_EQ(this->blob_top_2_->height(), 2);
  EXPECT_EQ(this->blob_top_2_->width(), 1);
>>>>>>> upstream/master

}

TYPED_TEST(MultiConvolutionLayerTest, TestSimpleMultiConvolutionGroup) {
  // We will simply see if the convolution layer carries out averaging well.
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MultiConvolutionParameter* convolution_param =
      layer_param.mutable_multi_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
<<<<<<< HEAD
  convolution_param->set_num_output(4*6);
  convolution_param->set_group(6);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  //convolution_param->mutable_bias_filler()->set_type("constant");
  //convolution_param->mutable_bias_filler()->set_value(0.2);
=======
  convolution_param->set_num_output(4*5);
  convolution_param->set_group(5);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("constant");
  convolution_param->mutable_bias_filler()->set_value(0.1);
>>>>>>> upstream/master
  shared_ptr<Layer<Dtype> > layer(
      new MultiConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Check against reference convolution.
  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_conv(this->blob_bottom_, convolution_param, layer->blobs(),
      this->MakeReferenceTop(this->blob_top_));
  top_data = this->blob_top_->cpu_data();
  ref_top_data = this->ref_blob_top_->cpu_data();
  for (int i = 0; i < this->blob_top_->count(); ++i) {
<<<<<<< HEAD
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4)<<i;
=======
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
>>>>>>> upstream/master
  }
}


TYPED_TEST(MultiConvolutionLayerTest, TestGradientGroup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  MultiConvolutionParameter* convolution_param =
      layer_param.mutable_multi_convolution_param();
  convolution_param->set_kernel_size(3);
  convolution_param->set_stride(2);
<<<<<<< HEAD
  convolution_param->set_num_output(4*6);
  convolution_param->set_group(6);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  MultiConvolutionLayer<Dtype> layer(layer_param);
=======
  convolution_param->set_num_output(4*5);
  convolution_param->set_group(5);
  convolution_param->mutable_weight_filler()->set_type("gaussian");
  convolution_param->mutable_bias_filler()->set_type("gaussian");
  ConvolutionLayer<Dtype> layer(layer_param);
>>>>>>> upstream/master
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}

<<<<<<< HEAD
}
=======
>>>>>>> upstream/master
