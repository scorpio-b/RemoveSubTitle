#include "remove_subtitle/subtitle_detector.hpp"

#include <gtest/gtest.h>

TEST(SubtitleDetectorTest, DetectsBrightSubtitleLikeRegionInsideRoi) {
    cv::Mat frame(120, 240, CV_8UC3, cv::Scalar(30, 30, 30));
    cv::rectangle(frame, cv::Rect(80, 80, 60, 18), cv::Scalar(240, 240, 240), cv::FILLED);

    const cv::Rect roi(40, 60, 160, 40);
    const remove_subtitle::DetectionResult detection = remove_subtitle::DetectSubtitleMask(frame, roi);

    EXPECT_GT(cv::countNonZero(detection.mask(roi)), 0);
    EXPECT_FALSE(detection.boxes.empty());
}
