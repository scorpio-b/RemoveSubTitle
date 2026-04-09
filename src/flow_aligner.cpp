#include "remove_subtitle/flow_aligner.hpp"

#include <algorithm>

namespace remove_subtitle {

double PatchDifference(
    const cv::Mat& a,
    const cv::Mat& b,
    const cv::Point& pt,
    int radius
) {
    const int x0 = std::max(0, pt.x - radius);
    const int y0 = std::max(0, pt.y - radius);
    const int x1 = std::min(a.cols - 1, pt.x + radius);
    const int y1 = std::min(a.rows - 1, pt.y + radius);
    const cv::Rect roi(x0, y0, x1 - x0 + 1, y1 - y0 + 1);
    cv::Mat diff;
    cv::absdiff(a(roi), b(roi), diff);
    const cv::Scalar mean_diff = cv::mean(diff);
    return (mean_diff[0] + mean_diff[1] + mean_diff[2]) / 3.0;
}

cv::Mat WarpReferenceToCurrent(const cv::Mat& current_bgr, const cv::Mat& reference_bgr) {
    cv::Mat current_gray;
    cv::Mat reference_gray;
    cv::cvtColor(current_bgr, current_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(reference_bgr, reference_gray, cv::COLOR_BGR2GRAY);

    cv::Mat flow;
    cv::calcOpticalFlowFarneback(
        reference_gray,
        current_gray,
        flow,
        0.5,
        3,
        21,
        5,
        7,
        1.5,
        0
    );

    cv::Mat map_x(current_gray.size(), CV_32FC1);
    cv::Mat map_y(current_gray.size(), CV_32FC1);
    for (int y = 0; y < current_gray.rows; ++y) {
        for (int x = 0; x < current_gray.cols; ++x) {
            const cv::Point2f delta = flow.at<cv::Point2f>(y, x);
            map_x.at<float>(y, x) = static_cast<float>(x - delta.x);
            map_y.at<float>(y, x) = static_cast<float>(y - delta.y);
        }
    }

    cv::Mat warped;
    cv::remap(reference_bgr, warped, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_REPLICATE);
    return warped;
}

}  // namespace remove_subtitle
