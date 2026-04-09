#pragma once

#include <opencv2/opencv.hpp>

namespace remove_subtitle {

cv::Rect ClampRect(const cv::Rect& rect, const cv::Size& frame_size);

}  // namespace remove_subtitle
