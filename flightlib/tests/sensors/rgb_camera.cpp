#include <gtest/gtest.h>

#include <limits>
#include <vector>

#include "flightlib/sensors/rgb_camera.hpp"

using namespace flightlib;

TEST(RGBCamera, DefaultParameters) {
  RGBCamera camera;

  EXPECT_EQ(camera.getWidth(), 720);
  EXPECT_EQ(camera.getHeight(), 480);
  EXPECT_EQ(camera.getChannels(), 3);
  EXPECT_FLOAT_EQ(camera.getFOV(), 70.0);
  EXPECT_FLOAT_EQ(camera.getDepthScale(), 0.2);

  EXPECT_EQ(camera.getEnabledLayers(),
            (std::vector<bool>{false, false, false}));
}

TEST(RGBCamera, DefaultRelativePose) {
  RGBCamera camera;

  EXPECT_TRUE(camera.getRelPose().isApprox(
    Matrix<4, 4>::Identity()));
}

TEST(RGBCamera, SetRelativePose) {
  RGBCamera camera;

  Vector<3> position;
  position << 0.1, -0.2, 0.3;

  Matrix<3, 3> rotation;
  rotation << 0, -1, 0,
              1,  0, 0,
              0,  0, 1;

  ASSERT_TRUE(camera.setRelPose(position, rotation));

  Matrix<4, 4> expected = Matrix<4, 4>::Identity();
  expected.block<3, 3>(0, 0) = rotation;
  expected.block<3, 1>(0, 3) = position;

  EXPECT_TRUE(camera.getRelPose().isApprox(expected));
}

TEST(RGBCamera, InvalidRelativePose) {
  RGBCamera camera;

  Vector<3> position = Vector<3>::Zero();
  Matrix<3, 3> rotation = Matrix<3, 3>::Identity();

  position(0) = std::numeric_limits<Scalar>::quiet_NaN();
  EXPECT_FALSE(camera.setRelPose(position, rotation));

  position.setZero();
  rotation(0, 0) = std::numeric_limits<Scalar>::infinity();
  EXPECT_FALSE(camera.setRelPose(position, rotation));

  EXPECT_TRUE(camera.getRelPose().isApprox(
    Matrix<4, 4>::Identity()));
}

TEST(RGBCamera, ImageDimensions) {
  RGBCamera camera;

  EXPECT_TRUE(camera.setWidth(640));
  EXPECT_TRUE(camera.setHeight(360));

  EXPECT_FALSE(camera.setWidth(0));
  EXPECT_FALSE(camera.setWidth(-1));
  EXPECT_FALSE(camera.setHeight(0));
  EXPECT_FALSE(camera.setHeight(-1));

  EXPECT_EQ(camera.getWidth(), 640);
  EXPECT_EQ(camera.getHeight(), 360);
}

TEST(RGBCamera, FieldOfView) {
  RGBCamera camera;

  EXPECT_TRUE(camera.setFOV(90.0));

  EXPECT_FALSE(camera.setFOV(0.0));
  EXPECT_FALSE(camera.setFOV(-10.0));
  EXPECT_FALSE(camera.setFOV(180.0));
  EXPECT_FALSE(camera.setFOV(200.0));
  EXPECT_FALSE(camera.setFOV(
    std::numeric_limits<Scalar>::quiet_NaN()));
  EXPECT_FALSE(camera.setFOV(
    std::numeric_limits<Scalar>::infinity()));

  EXPECT_FLOAT_EQ(camera.getFOV(), 90.0);
}

TEST(RGBCamera, DepthScale) {
  RGBCamera camera;

  EXPECT_TRUE(camera.setDepthScale(0.0));
  EXPECT_TRUE(camera.setDepthScale(1.0));
  EXPECT_TRUE(camera.setDepthScale(0.5));

  EXPECT_FALSE(camera.setDepthScale(-0.1));
  EXPECT_FALSE(camera.setDepthScale(1.1));
  EXPECT_FALSE(camera.setDepthScale(
    std::numeric_limits<Scalar>::quiet_NaN()));
  EXPECT_FALSE(camera.setDepthScale(
    std::numeric_limits<Scalar>::infinity()));

  EXPECT_FLOAT_EQ(camera.getDepthScale(), 0.5);
}

TEST(RGBCamera, PostProcessing) {
  RGBCamera camera;

  ASSERT_TRUE(camera.setPostProcesscing(
    {true, false, true}));

  EXPECT_EQ(camera.getEnabledLayers(),
            (std::vector<bool>{true, false, true}));

  EXPECT_FALSE(camera.setPostProcesscing({true}));
  EXPECT_FALSE(camera.setPostProcesscing(
    {true, true, true, true}));

  EXPECT_EQ(camera.getEnabledLayers(),
            (std::vector<bool>{true, false, true}));
}

TEST(RGBCamera, IndividualLayerSwitches) {
  RGBCamera camera;

  camera.enableDepth(true);
  EXPECT_EQ(camera.getEnabledLayers(),
            (std::vector<bool>{true, false, false}));

  camera.enableSegmentation(true);
  EXPECT_EQ(camera.getEnabledLayers(),
            (std::vector<bool>{true, true, false}));

  camera.enableOpticalFlow(true);
  EXPECT_EQ(camera.getEnabledLayers(),
            (std::vector<bool>{true, true, true}));

  camera.enableDepth(false);
  EXPECT_EQ(camera.getEnabledLayers(),
            (std::vector<bool>{false, true, true}));

  camera.enableSegmentation(false);
  EXPECT_EQ(camera.getEnabledLayers(),
            (std::vector<bool>{false, false, true}));

  camera.enableOpticalFlow(false);
  EXPECT_EQ(camera.getEnabledLayers(),
            (std::vector<bool>{false, false, false}));
}

TEST(RGBCamera, EmptyImageQueues) {
  RGBCamera camera;
  cv::Mat image;

  EXPECT_FALSE(camera.getRGBImage(image));
  EXPECT_FALSE(camera.getDepthMap(image));
  EXPECT_FALSE(camera.getSegmentation(image));
  EXPECT_FALSE(camera.getOpticalFlow(image));
}

TEST(RGBCamera, LatestImageQueue) {
  RGBCamera camera;

  using Getter = bool (RGBCamera::*)(cv::Mat&);

  const Getter getters[] = {
    &RGBCamera::getRGBImage,
    &RGBCamera::getDepthMap,
    &RGBCamera::getSegmentation,
    &RGBCamera::getOpticalFlow
  };

  for (int layer = 0; layer < 4; ++layer) {
    SCOPED_TRACE(layer);

    const int type = layer == 1 ? CV_32FC1 : CV_8UC3;

    for (int frame = 1; frame <= 10; ++frame) {
      cv::Mat image(
        4, 6, type, cv::Scalar::all(frame));

      ASSERT_TRUE(camera.feedImageQueue(layer, image));
    }

    cv::Mat output;
    ASSERT_TRUE((camera.*getters[layer])(output));

    EXPECT_EQ(output.rows, 4);
    EXPECT_EQ(output.cols, 6);
    EXPECT_EQ(output.type(), type);

    cv::Mat expected(
      4, 6, type, cv::Scalar::all(10));

    EXPECT_DOUBLE_EQ(
      cv::norm(output, expected, cv::NORM_INF), 0.0);

    EXPECT_FALSE((camera.*getters[layer])(output));
  }
}

TEST(RGBCamera, IndependentImageQueues) {
  RGBCamera camera;

  cv::Mat rgb(4, 6, CV_8UC3, cv::Scalar::all(10));
  cv::Mat depth(4, 6, CV_32FC1, cv::Scalar::all(20));
  cv::Mat segmentation(4, 6, CV_8UC3, cv::Scalar::all(30));
  cv::Mat flow(4, 6, CV_8UC3, cv::Scalar::all(40));

  ASSERT_TRUE(camera.feedImageQueue(0, rgb));
  ASSERT_TRUE(camera.feedImageQueue(1, depth));
  ASSERT_TRUE(camera.feedImageQueue(2, segmentation));
  ASSERT_TRUE(camera.feedImageQueue(3, flow));

  cv::Mat output;

  ASSERT_TRUE(camera.getRGBImage(output));
  EXPECT_DOUBLE_EQ(cv::norm(output, rgb, cv::NORM_INF), 0.0);

  ASSERT_TRUE(camera.getDepthMap(output));
  EXPECT_DOUBLE_EQ(cv::norm(output, depth, cv::NORM_INF), 0.0);

  ASSERT_TRUE(camera.getSegmentation(output));
  EXPECT_DOUBLE_EQ(
    cv::norm(output, segmentation, cv::NORM_INF), 0.0);

  ASSERT_TRUE(camera.getOpticalFlow(output));
  EXPECT_DOUBLE_EQ(cv::norm(output, flow, cv::NORM_INF), 0.0);
}

TEST(RGBCamera, InvalidImageLayer) {
  RGBCamera camera;

  cv::Mat image(4, 6, CV_8UC3, cv::Scalar::all(1));

  EXPECT_FALSE(camera.feedImageQueue(-1, image));
  EXPECT_FALSE(camera.feedImageQueue(4, image));
  EXPECT_FALSE(camera.feedImageQueue(100, image));

  cv::Mat output;

  EXPECT_FALSE(camera.getRGBImage(output));
  EXPECT_FALSE(camera.getDepthMap(output));
  EXPECT_FALSE(camera.getSegmentation(output));
  EXPECT_FALSE(camera.getOpticalFlow(output));

  // Verify that invalid layers do not leave the mutex locked.
  ASSERT_TRUE(camera.feedImageQueue(0, image));
  EXPECT_TRUE(camera.getRGBImage(output));
}
