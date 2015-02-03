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

template <typename TypeParam>
class RecursiveOnceTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;
protected:
	RecursiveOnceTest()
	// num: 1
	// channels: 5x10 ( frames x feature maps )
	// height, width = 10, 10
	: blob_bottom_(new Blob<Dtype>(1, 64, 3, 3)), 
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
    for (int i = 0; i < 64*9; i ++) {
      //blob_bottom_->mutable_cpu_data()[i] = 0.1;
    }
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
}

virtual ~RecursiveOnceTest() {
	delete blob_bottom_;
  delete blob_top_;
}

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(RecursiveOnceTest, TestDtypesAndDevices);


TYPED_TEST(RecursiveOnceTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  RecursiveOnceParameter* recursive_once_param = 
    layer_param.mutable_recursive_once_param();
  recursive_once_param->set_assemble_size(1); //2
  recursive_once_param->set_stride(2);
  recursive_once_param->set_num_uv(2);
  recursive_once_param->set_group(8); 
  recursive_once_param->set_bias_term(true); 
  recursive_once_param->add_relative_position(0);
  recursive_once_param->add_relative_position(1);

  shared_ptr<Layer<Dtype> > layer (
    new RecursiveOnceLayer<Dtype>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 32);
  EXPECT_EQ(this->blob_top_->height(), 3);
  EXPECT_EQ(this->blob_top_->width(), 3);
}


TYPED_TEST(RecursiveOnceTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  RecursiveOnceParameter* recursive_once_param = 
  	layer_param.mutable_recursive_once_param();

  recursive_once_param->set_group(8);
  recursive_once_param->set_assemble_size(2); //4
  recursive_once_param->set_stride(4);
  recursive_once_param->set_num_uv(3);
  recursive_once_param->add_relative_position(0);
  recursive_once_param->add_relative_position(2);
  recursive_once_param->add_relative_position(3);
  //recursive_once_param->set_bias_term(true); 
  recursive_once_param->mutable_weight_filler()->set_type("gaussian");
  //recursive_once_param->mutable_bias_filler()->set_type("gaussian");
  RecursiveOnceLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
      &(this->blob_top_vec_));
  }
 }


























