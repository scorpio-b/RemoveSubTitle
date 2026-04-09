#include "remove_subtitle/region_restorer.hpp"

#include <gtest/gtest.h>

TEST(RestorerTest, SpatialFallbackUsesNearbyUnmaskedPixels) {
    cv::Mat frame(7, 7, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat mask(7, 7, CV_8UC1, cv::Scalar(0));

    frame.at<cv::Vec3b>(3, 2) = cv::Vec3b(10, 20, 30);
    frame.at<cv::Vec3b>(3, 4) = cv::Vec3b(20, 30, 40);
    frame.at<cv::Vec3b>(2, 3) = cv::Vec3b(30, 40, 50);
    frame.at<cv::Vec3b>(4, 3) = cv::Vec3b(40, 50, 60);
    mask.at<uchar>(3, 3) = 255;

    const cv::Vec3b sample = remove_subtitle::SpatialFallbackSample(frame, mask, 3, 3);

    EXPECT_GT(sample[0], 0);
    EXPECT_GT(sample[1], 0);
    EXPECT_GT(sample[2], 0);
}
