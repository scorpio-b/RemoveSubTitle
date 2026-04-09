#include "remove_subtitle/geometry.hpp"

namespace remove_subtitle {

cv::Rect ClampRect(const cv::Rect& rect, const cv::Size& frame_size) {
    return rect & cv::Rect(0, 0, frame_size.width, frame_size.height);
}

}  // namespace remove_subtitle
