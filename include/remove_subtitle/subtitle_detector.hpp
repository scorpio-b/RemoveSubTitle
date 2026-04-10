#pragma once

#include <opencv2/opencv.hpp>

#include <vector>

namespace remove_subtitle {

struct DetectionResult {
    cv::Mat mask;
    cv::Mat repair_mask;
    cv::Mat expanded_outline_mask;
    cv::Mat ocr_mask;
    cv::Mat band_mask;
    std::vector<cv::Rect> boxes;
    int masked_pixels = 0;
};

DetectionResult DetectSubtitleMask(
    const cv::Mat& frame,
    const cv::Rect& roi_rect,
    int repair_expand
);

}  // namespace remove_subtitle
