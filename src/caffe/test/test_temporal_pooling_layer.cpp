#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include <iostream>

namespace caffe {

template <typename TypeParam>
class TemporalPoolingLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  TemporalPoolingLayerTest()
      : blob_bottom_(new Blob<Dtype>()),
        blob_top_(new Blob<Dtype>()),
        blob_top_mask_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    Caffe::set_random_seed(1701);
    blob_bottom_->Reshape(2, 15, 2, 2);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~TemporalPoolingLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
    delete blob_top_mask_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const blob_top_mask_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  // Test for 2x 2 square pooling layer
  void TestForwardSquare() {
    LayerParameter layer_param;
    TemporalPoolingParameter* pooling_param = layer_param.mutable_temporal_pooling_param();
    pooling_param->set_kernel_size(3);
    pooling_param->set_group(5);
    pooling_param->set_stride(2);
    //pooling_param->set_pool(TemporalPoolingParameter_PoolMethod_MAX);
    const int num = 2;
    const int channels = 15; // 3x5
    blob_bottom_->Reshape(num, channels, 2, 2);
    // Input: 2x 2 channels of:
    //     [1 2 5 2 3]
    //     [9 4 1 4 8]
    //     [1 2 5 2 3]
    for (int i = 0; i < num; i += 60) {
      blob_bottom_->mutable_cpu_data()[i +  0] = 1;
      blob_bottom_->mutable_cpu_data()[i +  1] = 2;
      blob_bottom_->mutable_cpu_data()[i +  2] = 3;
      blob_bottom_->mutable_cpu_data()[i +  3] = 4;
      blob_bottom_->mutable_cpu_data()[i +  4] = 5;
      blob_bottom_->mutable_cpu_data()[i +  5] = 6;
      blob_bottom_->mutable_cpu_data()[i +  6] = 7;
      blob_bottom_->mutable_cpu_data()[i +  7] = 8;
      blob_bottom_->mutable_cpu_data()[i +  8] = 9;
      blob_bottom_->mutable_cpu_data()[i +  9] = 10;
      blob_bottom_->mutable_cpu_data()[i + 10] = 11;
      blob_bottom_->mutable_cpu_data()[i + 11] = 12;

      blob_bottom_->mutable_cpu_data()[i + 12] = 3;
      blob_bottom_->mutable_cpu_data()[i + 13] = 1;
      blob_bottom_->mutable_cpu_data()[i + 14] = 2;
      blob_bottom_->mutable_cpu_data()[i + 15] = 5;
      blob_bottom_->mutable_cpu_data()[i + 16] = 5;
      blob_bottom_->mutable_cpu_data()[i + 17] = 4;
      blob_bottom_->mutable_cpu_data()[i + 18] = 2;
      blob_bottom_->mutable_cpu_data()[i + 19] = 3;
      blob_bottom_->mutable_cpu_data()[i + 20] = 8;
      blob_bottom_->mutable_cpu_data()[i + 21] = 7;
      blob_bottom_->mutable_cpu_data()[i + 22] = 15;
      blob_bottom_->mutable_cpu_data()[i + 23] = 2;

      blob_bottom_->mutable_cpu_data()[i + 24] = -2;
      blob_bottom_->mutable_cpu_data()[i + 25] = 5;
      blob_bottom_->mutable_cpu_data()[i + 26] = 7;
      blob_bottom_->mutable_cpu_data()[i + 27] = -3;
      blob_bottom_->mutable_cpu_data()[i + 28] = 4;
      blob_bottom_->mutable_cpu_data()[i + 29] = 5;
      blob_bottom_->mutable_cpu_data()[i + 30] = 8;
      blob_bottom_->mutable_cpu_data()[i + 31] = -2;
      blob_bottom_->mutable_cpu_data()[i + 32] = 7;
      blob_bottom_->mutable_cpu_data()[i + 33] = 3;
      blob_bottom_->mutable_cpu_data()[i + 34] = 7;
      blob_bottom_->mutable_cpu_data()[i + 35] = 8;

      blob_bottom_->mutable_cpu_data()[i + 36] = -3;
      blob_bottom_->mutable_cpu_data()[i + 37] = 5;
      blob_bottom_->mutable_cpu_data()[i + 38] = 7;
      blob_bottom_->mutable_cpu_data()[i + 39] = 4;
      blob_bottom_->mutable_cpu_data()[i + 40] = 8;
      blob_bottom_->mutable_cpu_data()[i + 41] = 5;
      blob_bottom_->mutable_cpu_data()[i + 42] = 2;
      blob_bottom_->mutable_cpu_data()[i + 43] = 1;
      blob_bottom_->mutable_cpu_data()[i + 44] = 9;
      blob_bottom_->mutable_cpu_data()[i + 45] = 3;
      blob_bottom_->mutable_cpu_data()[i + 46] = 8;
      blob_bottom_->mutable_cpu_data()[i + 47] = 10;

      blob_bottom_->mutable_cpu_data()[i + 48] = 7;
      blob_bottom_->mutable_cpu_data()[i + 49] = 4;
      blob_bottom_->mutable_cpu_data()[i + 50] = 2;
      blob_bottom_->mutable_cpu_data()[i + 51] = 5;
      blob_bottom_->mutable_cpu_data()[i + 52] = 3;
      blob_bottom_->mutable_cpu_data()[i + 53] = 9;
      blob_bottom_->mutable_cpu_data()[i + 54] = 7;
      blob_bottom_->mutable_cpu_data()[i + 55] = 6;
      blob_bottom_->mutable_cpu_data()[i + 56] = 1;
      blob_bottom_->mutable_cpu_data()[i + 57] = 8;
      blob_bottom_->mutable_cpu_data()[i + 58] = 5;
      blob_bottom_->mutable_cpu_data()[i + 59] = 2;
    }
    TemporalPoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, &blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), 6);
    EXPECT_EQ(blob_top_->height(), 2);
    EXPECT_EQ(blob_top_->width(), 2);
    //if (blob_top_vec_.size() > 1) {
    //  EXPECT_EQ(blob_top_mask_->num(), num);
    //  EXPECT_EQ(blob_top_mask_->channels(), 9);
    //  EXPECT_EQ(blob_top_mask_->height(), 2);
    //  EXPECT_EQ(blob_top_mask_->width(), 2);
    //}
    layer.Forward(blob_bottom_vec_, &blob_top_vec_);
    // Expected output: 2x 2 channels of:
    //     [9 5 5 8]
    //     [9 5 5 8]
    for (int i = 0; i < num; i += 24) {
      EXPECT_EQ(blob_top_->cpu_data()[i + 0], 3);
      EXPECT_EQ(blob_top_->cpu_data()[i + 1], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 2], 7);
      EXPECT_EQ(blob_top_->cpu_data()[i + 3], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 4], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 5], 6);
      EXPECT_EQ(blob_top_->cpu_data()[i + 6], 8);
      EXPECT_EQ(blob_top_->cpu_data()[i + 7], 8);
      EXPECT_EQ(blob_top_->cpu_data()[i + 8], 9);
      EXPECT_EQ(blob_top_->cpu_data()[i + 9], 10);
      EXPECT_EQ(blob_top_->cpu_data()[i + 10], 15);
      EXPECT_EQ(blob_top_->cpu_data()[i + 11], 12);
      EXPECT_EQ(blob_top_->cpu_data()[i + 12], 7);
      EXPECT_EQ(blob_top_->cpu_data()[i + 13], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 14], 7);
      EXPECT_EQ(blob_top_->cpu_data()[i + 15], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 16], 8);
      EXPECT_EQ(blob_top_->cpu_data()[i + 17], 9);
      EXPECT_EQ(blob_top_->cpu_data()[i + 18], 8);
      EXPECT_EQ(blob_top_->cpu_data()[i + 19], 6);
      EXPECT_EQ(blob_top_->cpu_data()[i + 20], 9);
      EXPECT_EQ(blob_top_->cpu_data()[i + 21], 8);
      EXPECT_EQ(blob_top_->cpu_data()[i + 22], 8);
      EXPECT_EQ(blob_top_->cpu_data()[i + 23], 10);
    }
    //if (blob_top_vec_.size() > 1) {
    //  // Expected mask output: 2x 2 channels of:
    //  //     [5  2  2 9]
    //  //     [5 12 12 9]
    //  for (int i = 0; i < 8 * num * channels; i += 8) {
    //    EXPECT_EQ(blob_top_mask_->cpu_data()[i + 0],  5);
    //    EXPECT_EQ(blob_top_mask_->cpu_data()[i + 1],  2);
    //    EXPECT_EQ(blob_top_mask_->cpu_data()[i + 2],  2);
    //    EXPECT_EQ(blob_top_mask_->cpu_data()[i + 3],  9);
    //    EXPECT_EQ(blob_top_mask_->cpu_data()[i + 4],  5);
    //    EXPECT_EQ(blob_top_mask_->cpu_data()[i + 5], 12);
    //    EXPECT_EQ(blob_top_mask_->cpu_data()[i + 6], 12);
    //    EXPECT_EQ(blob_top_mask_->cpu_data()[i + 7],  9);
    //  }
    //}
  }
  // Test for 2x 2 square pooling layer
  void TestForwardSquarePad() {
    LayerParameter layer_param;
    TemporalPoolingParameter* pooling_param = layer_param.mutable_temporal_pooling_param();
    pooling_param->set_kernel_size(2);
    pooling_param->set_group(5);
    pooling_param->set_pad(1);
    pooling_param->set_stride(2);
    //pooling_param->set_pool(TemporalPoolingParameter_PoolMethod_MAX);
    const int num = 2;
    const int channels = 15; // 3x5
    blob_bottom_->Reshape(num, channels, 2, 2);
    // Input: 2x 2 channels of:
    //     [1 2 5 2 3]
    //     [9 4 1 4 8]
    //     [1 2 5 2 3]
    for (int i = 0; i < num; i += 60) {
      blob_bottom_->mutable_cpu_data()[i +  0] = 1;
      blob_bottom_->mutable_cpu_data()[i +  1] = 2;
      blob_bottom_->mutable_cpu_data()[i +  2] = 3;
      blob_bottom_->mutable_cpu_data()[i +  3] = 4;
      blob_bottom_->mutable_cpu_data()[i +  4] = 5;
      blob_bottom_->mutable_cpu_data()[i +  5] = 6;
      blob_bottom_->mutable_cpu_data()[i +  6] = 7;
      blob_bottom_->mutable_cpu_data()[i +  7] = 8;
      blob_bottom_->mutable_cpu_data()[i +  8] = 9;
      blob_bottom_->mutable_cpu_data()[i +  9] = 10;
      blob_bottom_->mutable_cpu_data()[i + 10] = 11;
      blob_bottom_->mutable_cpu_data()[i + 11] = 12;

      blob_bottom_->mutable_cpu_data()[i + 12] = 3;
      blob_bottom_->mutable_cpu_data()[i + 13] = 1;
      blob_bottom_->mutable_cpu_data()[i + 14] = 2;
      blob_bottom_->mutable_cpu_data()[i + 15] = 5;
      blob_bottom_->mutable_cpu_data()[i + 16] = 5;
      blob_bottom_->mutable_cpu_data()[i + 17] = 4;
      blob_bottom_->mutable_cpu_data()[i + 18] = 2;
      blob_bottom_->mutable_cpu_data()[i + 19] = 3;
      blob_bottom_->mutable_cpu_data()[i + 20] = 8;
      blob_bottom_->mutable_cpu_data()[i + 21] = 7;
      blob_bottom_->mutable_cpu_data()[i + 22] = 15;
      blob_bottom_->mutable_cpu_data()[i + 23] = 2;

      blob_bottom_->mutable_cpu_data()[i + 24] = -2;
      blob_bottom_->mutable_cpu_data()[i + 25] = 5;
      blob_bottom_->mutable_cpu_data()[i + 26] = 7;
      blob_bottom_->mutable_cpu_data()[i + 27] = -3;
      blob_bottom_->mutable_cpu_data()[i + 28] = 4;
      blob_bottom_->mutable_cpu_data()[i + 29] = 5;
      blob_bottom_->mutable_cpu_data()[i + 30] = 8;
      blob_bottom_->mutable_cpu_data()[i + 31] = -2;
      blob_bottom_->mutable_cpu_data()[i + 32] = 7;
      blob_bottom_->mutable_cpu_data()[i + 33] = 3;
      blob_bottom_->mutable_cpu_data()[i + 34] = 7;
      blob_bottom_->mutable_cpu_data()[i + 35] = 8;

      blob_bottom_->mutable_cpu_data()[i + 36] = -3;
      blob_bottom_->mutable_cpu_data()[i + 37] = 5;
      blob_bottom_->mutable_cpu_data()[i + 38] = 7;
      blob_bottom_->mutable_cpu_data()[i + 39] = 4;
      blob_bottom_->mutable_cpu_data()[i + 40] = 8;
      blob_bottom_->mutable_cpu_data()[i + 41] = 5;
      blob_bottom_->mutable_cpu_data()[i + 42] = 2;
      blob_bottom_->mutable_cpu_data()[i + 43] = 1;
      blob_bottom_->mutable_cpu_data()[i + 44] = 9;
      blob_bottom_->mutable_cpu_data()[i + 45] = 3;
      blob_bottom_->mutable_cpu_data()[i + 46] = 8;
      blob_bottom_->mutable_cpu_data()[i + 47] = 10;

      blob_bottom_->mutable_cpu_data()[i + 48] = 7;
      blob_bottom_->mutable_cpu_data()[i + 49] = 4;
      blob_bottom_->mutable_cpu_data()[i + 50] = 2;
      blob_bottom_->mutable_cpu_data()[i + 51] = 5;
      blob_bottom_->mutable_cpu_data()[i + 52] = 3;
      blob_bottom_->mutable_cpu_data()[i + 53] = 9;
      blob_bottom_->mutable_cpu_data()[i + 54] = 7;
      blob_bottom_->mutable_cpu_data()[i + 55] = 6;
      blob_bottom_->mutable_cpu_data()[i + 56] = 1;
      blob_bottom_->mutable_cpu_data()[i + 57] = 8;
      blob_bottom_->mutable_cpu_data()[i + 58] = 5;
      blob_bottom_->mutable_cpu_data()[i + 59] = 2;
    }
    TemporalPoolingLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec_, &blob_top_vec_);
    EXPECT_EQ(blob_top_->num(), num);
    EXPECT_EQ(blob_top_->channels(), 9);
    EXPECT_EQ(blob_top_->height(), 2);
    EXPECT_EQ(blob_top_->width(), 2);
    //if (blob_top_vec_.size() > 1) {
    //  EXPECT_EQ(blob_top_mask_->num(), num);
    //  EXPECT_EQ(blob_top_mask_->channels(), 9);
    //  EXPECT_EQ(blob_top_mask_->height(), 2);
    //  EXPECT_EQ(blob_top_mask_->width(), 2);
    //}
    layer.Forward(blob_bottom_vec_, &blob_top_vec_);
    // Expected output: 2x 2 channels of:
    //     [9 5 5 8]
    //     [9 5 5 8]
    for (int i = 0; i < num; i += 36) {
      EXPECT_EQ(blob_top_->cpu_data()[i + 0], 1);
      EXPECT_EQ(blob_top_->cpu_data()[i + 1], 2);
      EXPECT_EQ(blob_top_->cpu_data()[i + 2], 3);
      EXPECT_EQ(blob_top_->cpu_data()[i + 3], 4);
      EXPECT_EQ(blob_top_->cpu_data()[i + 4], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 5], 6);
      EXPECT_EQ(blob_top_->cpu_data()[i + 6], 7);
      EXPECT_EQ(blob_top_->cpu_data()[i + 7], 8);
      EXPECT_EQ(blob_top_->cpu_data()[i + 8], 9);
      EXPECT_EQ(blob_top_->cpu_data()[i + 9], 10);
      EXPECT_EQ(blob_top_->cpu_data()[i + 10], 11);
      EXPECT_EQ(blob_top_->cpu_data()[i + 11], 12);
      EXPECT_EQ(blob_top_->cpu_data()[i + 12], 3);
      EXPECT_EQ(blob_top_->cpu_data()[i + 13], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 14], 7);
      EXPECT_EQ(blob_top_->cpu_data()[i + 15], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 16], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 17], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 18], 8);
      EXPECT_EQ(blob_top_->cpu_data()[i + 19], 3);
      EXPECT_EQ(blob_top_->cpu_data()[i + 20], 8);
      EXPECT_EQ(blob_top_->cpu_data()[i + 21], 7);
      EXPECT_EQ(blob_top_->cpu_data()[i + 22], 15);
      EXPECT_EQ(blob_top_->cpu_data()[i + 23], 8);
      EXPECT_EQ(blob_top_->cpu_data()[i + 24], 7);
      EXPECT_EQ(blob_top_->cpu_data()[i + 25], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 26], 7);
      EXPECT_EQ(blob_top_->cpu_data()[i + 27], 5);
      EXPECT_EQ(blob_top_->cpu_data()[i + 28], 8);
      EXPECT_EQ(blob_top_->cpu_data()[i + 29], 9);
      EXPECT_EQ(blob_top_->cpu_data()[i + 30], 7);
      EXPECT_EQ(blob_top_->cpu_data()[i + 31], 6);
      EXPECT_EQ(blob_top_->cpu_data()[i + 32], 9);
      EXPECT_EQ(blob_top_->cpu_data()[i + 33], 8);
      EXPECT_EQ(blob_top_->cpu_data()[i + 34], 8);
      EXPECT_EQ(blob_top_->cpu_data()[i + 35], 10);
    }
    //if (blob_top_vec_.size() > 1) {
    //  // Expected mask output: 2x 2 channels of:
    //  //     [5  2  2 9]
    //  //     [5 12 12 9]
    //  for (int i = 0; i < 8 * num * channels; i += 8) {
    //    EXPECT_EQ(blob_top_mask_->cpu_data()[i + 0],  5);
    //    EXPECT_EQ(blob_top_mask_->cpu_data()[i + 1],  2);
    //    EXPECT_EQ(blob_top_mask_->cpu_data()[i + 2],  2);
    //    EXPECT_EQ(blob_top_mask_->cpu_data()[i + 3],  9);
    //    EXPECT_EQ(blob_top_mask_->cpu_data()[i + 4],  5);
    //    EXPECT_EQ(blob_top_mask_->cpu_data()[i + 5], 12);
    //    EXPECT_EQ(blob_top_mask_->cpu_data()[i + 6], 12);
    //    EXPECT_EQ(blob_top_mask_->cpu_data()[i + 7],  9);
    //  }
    //}
  }
};

TYPED_TEST_CASE(TemporalPoolingLayerTest, TestDtypesAndDevices);

TYPED_TEST(TemporalPoolingLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  TemporalPoolingParameter* pooling_param = layer_param.mutable_temporal_pooling_param();
  pooling_param->set_kernel_size(2);
  pooling_param->set_stride(2);
  //pooling_param->set_pool(TemporalPoolingParameter_PoolMethod_MAX);
  pooling_param->set_group(5);
  TemporalPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  //EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->channels(), 9);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 2);
}

TYPED_TEST(TemporalPoolingLayerTest, TestSetupPadded) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  TemporalPoolingParameter* pooling_param = layer_param.mutable_temporal_pooling_param();
  pooling_param->set_kernel_size(2);
  pooling_param->set_stride(2);
  pooling_param->set_pad(1);
  pooling_param->set_group(5);
  //pooling_param->set_pool(TemporalPoolingParameter_PoolMethod_MAX);
  TemporalPoolingLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), 9);
  EXPECT_EQ(this->blob_top_->height(), 2);
  EXPECT_EQ(this->blob_top_->width(), 2);
}

/*
TYPED_TEST(TemporalPoolingLayerTest, PrintBackward) {
  LayerParameter layer_param;
  layer_param.set_kernelsize(3);
  layer_param.set_stride(2);
  layer_param.set_pool(TemporalPoolingParameter_PoolMethod_MAX);
  TemporalPoolingLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    cout << "bottom data " << i << " " << this->blob_bottom_->cpu_data()[i] << endl;
  }
  for (int i = 0; i < this->blob_top_->count(); ++i) {
    cout << "top data " << i << " " << this->blob_top_->cpu_data()[i] << endl;
  }

  for (int i = 0; i < this->blob_top_->count(); ++i) {
    this->blob_top_->mutable_cpu_diff()[i] = i;
  }
  layer.Backward(this->blob_top_vec_, true, &(this->blob_bottom_vec_));
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    cout << "bottom diff " << i << " " << this->blob_bottom_->cpu_diff()[i] << endl;
  }
}
*/

TYPED_TEST(TemporalPoolingLayerTest, TestForwardMax) {
  this->TestForwardSquare();
  //this->TestForwardRectHigh();
  //this->TestForwardRectWide();
}

//TYPED_TEST(TemporalPoolingLayerTest, TestForwardMaxTopMask) {
//  this->blob_top_vec_.push_back(this->blob_top_mask_);
//  this->TestForwardSquare();
//  this->TestForwardRectHigh();
//  this->TestForwardRectWide();
//}

TYPED_TEST(TemporalPoolingLayerTest, TestGradientMax) {
  typedef typename TypeParam::Dtype Dtype;
  for (int kernel= 2; kernel<= 3; kernel++) {
      LayerParameter layer_param;
      TemporalPoolingParameter* pooling_param = layer_param.mutable_temporal_pooling_param();
      pooling_param->set_kernel_size(kernel);
      pooling_param->set_stride(2);
      pooling_param->set_group(5);
      pooling_param->set_pad(0);
      pooling_param->set_pool(TemporalPoolingParameter_PoolMethod_MAX);
      TemporalPoolingLayer<Dtype> layer(layer_param);
      GradientChecker<Dtype> checker(1e-4, 1e-2);
      checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
          &(this->blob_top_vec_));
  }
}

TYPED_TEST(TemporalPoolingLayerTest, TestForwardMaxPadded) {
  this->TestForwardSquarePad();
  //this->TestForwardRectHigh();
  //this->TestForwardRectWide();
}

TYPED_TEST(TemporalPoolingLayerTest, TestGradientMaxPadded) {
  typedef typename TypeParam::Dtype Dtype;
  for (int kernel= 2; kernel<= 3; kernel++) {
      LayerParameter layer_param;
      TemporalPoolingParameter* pooling_param = layer_param.mutable_temporal_pooling_param();
      pooling_param->set_kernel_size(kernel);
      pooling_param->set_stride(2);
      pooling_param->set_group(5);
      pooling_param->set_pad(kernel - 1);
      pooling_param->set_pool(TemporalPoolingParameter_PoolMethod_MAX);
      TemporalPoolingLayer<Dtype> layer(layer_param);
      GradientChecker<Dtype> checker(1e-4, 1e-2);
      checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
          &(this->blob_top_vec_));
  }
}

//TYPED_TEST(TemporalPoolingLayerTest, TestGradientMaxTopMask) {
//  typedef typename TypeParam::Dtype Dtype;
//  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
//    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
//      LayerParameter layer_param;
//      TemporalPoolingParameter* pooling_param = layer_param.mutable_temporal_pooling_param();
//      pooling_param->set_kernel_h(kernel_h);
//      pooling_param->set_kernel_w(kernel_w);
//      pooling_param->set_stride(2);
//      pooling_param->set_pool(TemporalPoolingParameter_PoolMethod_MAX);
//      this->blob_top_vec_.push_back(this->blob_top_mask_);
//      TemporalPoolingLayer<Dtype> layer(layer_param);
//      GradientChecker<Dtype> checker(1e-4, 1e-2);
//      checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
//          &(this->blob_top_vec_));
//      this->blob_top_vec_.pop_back();
//    }
//  }
//}

//TYPED_TEST(TemporalPoolingLayerTest, TestForwardAve) {
//  typedef typename TypeParam::Dtype Dtype;
//  LayerParameter layer_param;
//  TemporalPoolingParameter* pooling_param = layer_param.mutable_temporal_pooling_param();
//  pooling_param->set_kernel_size(3);
//  pooling_param->set_stride(1);
//  pooling_param->set_pad(1);
//  pooling_param->set_pool(TemporalPoolingParameter_PoolMethod_AVE);
//  this->blob_bottom_->Reshape(1, 1, 3, 3);
//  FillerParameter filler_param;
//  filler_param.set_value(Dtype(2));
//  ConstantFiller<Dtype> filler(filler_param);
//  filler.Fill(this->blob_bottom_);
//  TemporalPoolingLayer<Dtype> layer(layer_param);
//  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
//  EXPECT_EQ(this->blob_top_->num(), 1);
//  EXPECT_EQ(this->blob_top_->channels(), 1);
//  EXPECT_EQ(this->blob_top_->height(), 3);
//  EXPECT_EQ(this->blob_top_->width(), 3);
//  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
//  Dtype epsilon = 1e-5;
//  EXPECT_NEAR(this->blob_top_->cpu_data()[0], 8.0 / 9, epsilon);
//  EXPECT_NEAR(this->blob_top_->cpu_data()[1], 4.0 / 3, epsilon);
//  EXPECT_NEAR(this->blob_top_->cpu_data()[2], 8.0 / 9, epsilon);
//  EXPECT_NEAR(this->blob_top_->cpu_data()[3], 4.0 / 3, epsilon);
//  EXPECT_NEAR(this->blob_top_->cpu_data()[4], 2.0    , epsilon);
//  EXPECT_NEAR(this->blob_top_->cpu_data()[5], 4.0 / 3, epsilon);
//  EXPECT_NEAR(this->blob_top_->cpu_data()[6], 8.0 / 9, epsilon);
//  EXPECT_NEAR(this->blob_top_->cpu_data()[7], 4.0 / 3, epsilon);
//  EXPECT_NEAR(this->blob_top_->cpu_data()[8], 8.0 / 9, epsilon);
//}

















//TYPED_TEST(TemporalPoolingLayerTest, TestGradientAve) {
//  typedef typename TypeParam::Dtype Dtype;
//  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
//    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
//      LayerParameter layer_param;
//      TemporalPoolingParameter* pooling_param = layer_param.mutable_temporal_pooling_param();
//      pooling_param->set_kernel_h(kernel_h);
//      pooling_param->set_kernel_w(kernel_w);
//      pooling_param->set_stride(2);
//      pooling_param->set_pool(TemporalPoolingParameter_PoolMethod_AVE);
//      TemporalPoolingLayer<Dtype> layer(layer_param);
//      GradientChecker<Dtype> checker(1e-2, 1e-2);
//      checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
//          &(this->blob_top_vec_));
//    }
//  }
//}
//
//TYPED_TEST(TemporalPoolingLayerTest, TestGradientAvePadded) {
//  typedef typename TypeParam::Dtype Dtype;
//  for (int kernel_h = 3; kernel_h <= 4; kernel_h++) {
//    for (int kernel_w = 3; kernel_w <= 4; kernel_w++) {
//      LayerParameter layer_param;
//      TemporalPoolingParameter* pooling_param = layer_param.mutable_temporal_pooling_param();
//      pooling_param->set_kernel_h(kernel_h);
//      pooling_param->set_kernel_w(kernel_w);
//      pooling_param->set_stride(2);
//      pooling_param->set_pad(2);
//      pooling_param->set_pool(TemporalPoolingParameter_PoolMethod_AVE);
//      TemporalPoolingLayer<Dtype> layer(layer_param);
//      GradientChecker<Dtype> checker(1e-2, 1e-2);
//      checker.CheckGradientExhaustive(&layer, &(this->blob_bottom_vec_),
//          &(this->blob_top_vec_));
//    }
//  }
//}

}  // namespace caffe
