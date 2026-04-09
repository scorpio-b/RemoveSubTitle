#pragma once

#include <opencv2/opencv.hpp>

#include <vector>

namespace remove_subtitle {

struct DetectionResult {
    cv::Mat mask;
    cv::Mat band_mask;
    std::vector<cv::Rect> boxes;
    int masked_pixels = 0;
};

DetectionResult DetectSubtitleMask(const cv::Mat& frame, const cv::Rect& roi_rect);

}  // namespace remove_subtitle
