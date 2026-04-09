#pragma once

#include <opencv2/opencv.hpp>

#include <vector>

namespace remove_subtitle {

cv::Vec3b SpatialFallbackSample(
    const cv::Mat& frame,
    const cv::Mat& mask,
    int y,
    int x
);

cv::Mat RestoreFrame(
    const std::vector<cv::Mat>& frames,
    const std::vector<cv::Mat>& masks,
    int index,
    int temporal_window,
    int patch_radius
);

}  // namespace remove_subtitle
