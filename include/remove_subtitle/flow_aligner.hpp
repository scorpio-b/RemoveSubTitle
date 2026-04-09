#pragma once

#include <opencv2/opencv.hpp>

namespace remove_subtitle {

double PatchDifference(
    const cv::Mat& a,
    const cv::Mat& b,
    const cv::Point& pt,
    int radius
);

cv::Mat WarpReferenceToCurrent(const cv::Mat& current_bgr, const cv::Mat& reference_bgr);

}  // namespace remove_subtitle
