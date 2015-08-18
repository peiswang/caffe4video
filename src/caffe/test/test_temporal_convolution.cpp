#include <cstring>
#include <vector>
#include <stdio.h>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#define MAX(a,b) ((a) > (b) ? (a) : (b)) 

namespace caffe {

template <typename Dtype>
void caffe_temporal_conv(const Blob<Dtype>* in, TemporalConvolutionParameter* param,
                const vector<shared_ptr<Blob<Dtype> > >& weights,
                Blob<Dtype> * out) {
  int group = param->group();
  int stride = param->stride();
  int num_output = param->num_output();
  int kernel_size = param->kernel_size();
  int pad = param->pad();

  int out_group = (group + 2 * pad - kernel_size) / stride + 1;

  int channels = in->channels();
  int vl = channels / group;

  CHECK_EQ(out_group, out->channels()/num_output);

  int height = in->height();
  int width = in->width();

  const Dtype* in_data = in->cpu_data();
  const Dtype* weight_data = weights[0]->cpu_data();
  Dtype* out_data = out->mutable_cpu_data();
  const Dtype* bias_data = NULL;

  if (param->bias_term()) {
    bias_data = weights[1]->cpu_data();
  }

  for (int n = 0 ; n < out->num(); ++n) {
    for (int g = 0; g < out_group ; ++g) {

      for (int v = 0; v < num_output; ++v) {
        for (int y = 0; y < height; ++y) {
          for (int x = 0; x < width; ++x) {
            Dtype tmp = 0.0;
            for (int ks=0; ks<kernel_size;ks++) {
              for (int vi=0;vi<vl;vi++)
              {
                 tmp += weight_data[weights[0]->offset(0, v, ks, vi)] * in_data[in->offset(n, g*stride*vl + ks*vl + vi, y, x)];
              }
            }
            
            if (param->bias_term()) {
              tmp += bias_data[weights[1]->offset(0,0,0, v)];
            }
            out_data[out->offset(n, g*num_output+v, y, x)] = tmp;
            //for(int as = 0; as < assemble_size; ++as) {
            //  Dtype tmp = 0.0;
            //  for (int uv = 0; uv < num_uv; ++uv) {
            //      for (int vw = 0; vw < vl; ++vw) {
            //        tmp += weight_data[weights[0]->offset(as, uv, v, vw)] * 
            //               in_data[in->offset(n, 
            //                       stride*g*vl+vl*param->relative_position(uv)+vw,
            //                       y, x)];
            //      }
            //  }
            //  if (param->bias_term()) {
            //    tmp += bias_data[weights[1]->offset(0,0,as, v)];
            //  }
            //  out_data[out->offset(n, g*vl+v, y, x)] = 
            //     MAX(out_data[out->offset(n, g*vl+v, y, x)], tmp);
            //}

          }
        }
      }
    }
  }
  
}

template <typename TypeParam>
class TemporalConvolutionTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
protected:
	TemporalConvolutionTest()
	// num: 1
	// channels: 8x12 ( frames x feature maps )
	// height, width = 3, 3 
	: blob_bottom_(new Blob<Dtype>(3, 96, 3, 3)), 
	  blob_top_(new Blob<Dtype>())
  {}
  virtual void SetUp() {
  	// fill the values
  	FillerParameter filler_param;
    filler_param.set_value(1.);
    //filler_param.set_std(1.);
    //filler_param.set_mean(0.);
  	GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    //for (int i = 0; i < 96*9; i ++) {
    //  blob_bottom_->mutable_cpu_data()[i] = 0.1;
    //}
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  
  virtual ~TemporalConvolutionTest() {
  	delete blob_bottom_;
    delete blob_top_;
  }
  
  virtual Blob<Dtype>* MakeReferenceTop(Blob<Dtype>* top) {
    this->ref_blob_top_.reset(new Blob<Dtype>());
    this->ref_blob_top_->ReshapeLike(*top);
    return this->ref_blob_top_.get();
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  shared_ptr<Blob<Dtype> > ref_blob_top_;
};

TYPED_TEST_CASE(TemporalConvolutionTest, TestDtypesAndDevices);


TYPED_TEST(TemporalConvolutionTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  TemporalConvolutionParameter* temporal_convolution_param = 
    layer_param.mutable_temporal_convolution_param();

  temporal_convolution_param->set_stride(2);
  temporal_convolution_param->set_group(8); 
  temporal_convolution_param->set_num_output(10);
  temporal_convolution_param->set_kernel_size(3);

  temporal_convolution_param->set_bias_term(true); 

  shared_ptr<Layer<Dtype> > layer (
    new TemporalConvolutionLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 3);
  EXPECT_EQ(this->blob_top_->channels(), 30);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 3);
}

TYPED_TEST(TemporalConvolutionTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  TemporalConvolutionParameter* temporal_convolution_param = 
    layer_param.mutable_temporal_convolution_param();

  temporal_convolution_param->set_stride(1);
  temporal_convolution_param->set_group(8); 
  temporal_convolution_param->set_num_output(10);
  temporal_convolution_param->set_kernel_size(3);

  temporal_convolution_param->mutable_weight_filler()->set_type("gaussian");
  //temporal_convolution_param->mutable_weight_filler()->set_value(0.3);
  temporal_convolution_param->mutable_bias_filler()->set_type("gaussian");
  //temporal_convolution_param->mutable_bias_filler()->set_value(0.1);

  shared_ptr<Layer<Dtype> > layer (
    new TemporalConvolutionLayer<Dtype>(layer_param));

  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));

  const Dtype* top_data;
  const Dtype* ref_top_data;
  caffe_temporal_conv(this->blob_bottom_, temporal_convolution_param, layer->blobs(),
                  this->MakeReferenceTop(this->blob_top_));
  ref_top_data = this->ref_blob_top_->cpu_data();

  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  top_data = this->blob_top_->cpu_data();

  for (int i = 0; i < this->blob_top_->count(); ++i) {
    EXPECT_NEAR(top_data[i], ref_top_data[i], 1e-4);
  }
  
}

TYPED_TEST(TemporalConvolutionTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  TemporalConvolutionParameter* temporal_convolution_param = 
  	layer_param.mutable_temporal_convolution_param();

  temporal_convolution_param->set_stride(1);
  temporal_convolution_param->set_group(8); 
  temporal_convolution_param->set_num_output(10);
  temporal_convolution_param->set_kernel_size(3);

  //temporal_convolution_param->set_bias_term(true); 
  temporal_convolution_param->mutable_weight_filler()->set_type("gaussian");
  temporal_convolution_param->mutable_bias_filler()->set_type("gaussian");

  TemporalConvolutionLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
}



}



