#include "remove_subtitle/subtitle_detector.hpp"

#include <algorithm>

namespace remove_subtitle {

namespace {

std::vector<cv::Rect> MergeNearbyBoxes(const std::vector<cv::Rect>& input_boxes) {
    std::vector<cv::Rect> merged = input_boxes;
    bool changed = true;
    while (changed) {
        changed = false;
        for (std::size_t i = 0; i < merged.size() && !changed; ++i) {
            for (std::size_t j = i + 1; j < merged.size(); ++j) {
                const cv::Rect inflated_i = merged[i] + cv::Size(18, 12);
                const cv::Rect inflated_j = merged[j] + cv::Size(18, 12);
                if ((inflated_i & inflated_j).area() == 0) {
                    continue;
                }

                merged[i] |= merged[j];
                merged.erase(merged.begin() + static_cast<long>(j));
                changed = true;
                break;
            }
        }
    }
    return merged;
}

std::vector<cv::Rect> SelectSubtitleLineBoxes(
    const std::vector<cv::Rect>& input_boxes,
    const cv::Size& roi_size
) {
    if (input_boxes.empty()) {
        return {};
    }

    struct Cluster {
        std::vector<cv::Rect> boxes;
        int min_center_y = 0;
        int max_center_y = 0;
        int total_width = 0;
    };

    std::vector<Cluster> clusters;
    for (const cv::Rect& box : input_boxes) {
        const int center_y = box.y + box.height / 2;
        bool assigned = false;
        for (Cluster& cluster : clusters) {
            if (std::abs(center_y - cluster.min_center_y) <= 18 ||
                std::abs(center_y - cluster.max_center_y) <= 18) {
                cluster.boxes.push_back(box);
                cluster.min_center_y = std::min(cluster.min_center_y, center_y);
                cluster.max_center_y = std::max(cluster.max_center_y, center_y);
                cluster.total_width += box.width;
                assigned = true;
                break;
            }
        }

        if (!assigned) {
            clusters.push_back(Cluster{{box}, center_y, center_y, box.width});
        }
    }

    auto score_cluster = [&roi_size](const Cluster& cluster) {
        int mean_center_y = 0;
        for (const cv::Rect& box : cluster.boxes) {
            mean_center_y += box.y + box.height / 2;
        }
        mean_center_y /= static_cast<int>(cluster.boxes.size());

        const int distance_to_bottom_band = std::abs(mean_center_y - static_cast<int>(roi_size.height * 0.72));
        return cluster.total_width - distance_to_bottom_band * 3;
    };

    auto best_it = std::max_element(
        clusters.begin(),
        clusters.end(),
        [&](const Cluster& lhs, const Cluster& rhs) {
            return score_cluster(lhs) < score_cluster(rhs);
        }
    );

    if (best_it == clusters.end()) {
        return {};
    }

    std::vector<cv::Rect> selected;
    for (const cv::Rect& box : best_it->boxes) {
        const int center_y = box.y + box.height / 2;
        if (center_y < static_cast<int>(roi_size.height * 0.45) ||
            center_y > static_cast<int>(roi_size.height * 0.92)) {
            continue;
        }
        selected.push_back(box);
    }

    return selected;
}

}  // namespace

DetectionResult DetectSubtitleMask(const cv::Mat& frame, const cv::Rect& roi_rect) {
    DetectionResult result;
    result.mask = cv::Mat(frame.size(), CV_8UC1, cv::Scalar(0));

    cv::Mat roi = frame(roi_rect);
    cv::Mat gray;
    cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);

    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(9, 9), 0.0);
    cv::Mat local_diff;
    cv::absdiff(gray, blurred, local_diff);
    cv::Mat contrast_mask;
    cv::threshold(local_diff, contrast_mask, 18, 255, cv::THRESH_BINARY);

    cv::Mat adaptive_mask;
    cv::adaptiveThreshold(
        gray,
        adaptive_mask,
        255,
        cv::ADAPTIVE_THRESH_GAUSSIAN_C,
        cv::THRESH_BINARY,
        31,
        -8
    );

    cv::Mat hsv;
    cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsv_channels;
    cv::split(hsv, hsv_channels);

    cv::Mat low_saturation;
    cv::threshold(hsv_channels[1], low_saturation, 110, 255, cv::THRESH_BINARY_INV);

    cv::Mat bright_value;
    cv::threshold(hsv_channels[2], bright_value, 85, 255, cv::THRESH_BINARY);

    cv::Mat dark_outline;
    cv::threshold(hsv_channels[2], dark_outline, 105, 255, cv::THRESH_BINARY_INV);

    cv::Mat bright_text_mask;
    cv::bitwise_and(contrast_mask, low_saturation, bright_text_mask);
    cv::bitwise_and(bright_text_mask, bright_value, bright_text_mask);

    cv::Mat outline_mask;
    cv::bitwise_and(contrast_mask, dark_outline, outline_mask);

    cv::Mat combined;
    cv::bitwise_or(bright_text_mask, outline_mask, combined);
    cv::bitwise_or(combined, adaptive_mask, combined);

    cv::Mat closed;
    cv::Mat close_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 5));
    cv::morphologyEx(combined, closed, cv::MORPH_CLOSE, close_kernel);

    cv::Mat dilated;
    cv::Mat dilate_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 3));
    cv::dilate(closed, dilated, dilate_kernel, cv::Point(-1, -1), 1);

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int count = cv::connectedComponentsWithStats(dilated, labels, stats, centroids, 8);

    std::vector<cv::Rect> boxes;
    for (int label = 1; label < count; ++label) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        const int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
        const int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
        const int left = stats.at<int>(label, cv::CC_STAT_LEFT);
        const int top = stats.at<int>(label, cv::CC_STAT_TOP);

        if (area < 24 || width < 8 || height < 8) {
            continue;
        }

        if (width > roi_rect.width - 10 || height > roi_rect.height - 10) {
            continue;
        }

        if (height > 42) {
            continue;
        }

        const int rect_x = std::max(0, left - 6);
        const int rect_y = std::max(0, top - 6);
        const cv::Rect component_rect(
            rect_x,
            rect_y,
            std::min(roi.cols - rect_x, width + 12),
            std::min(roi.rows - rect_y, height + 12)
        );
        boxes.push_back(component_rect);
    }

    boxes = SelectSubtitleLineBoxes(boxes, roi.size());
    boxes = MergeNearbyBoxes(boxes);

    cv::Mat roi_mask(roi.size(), CV_8UC1, cv::Scalar(0));
    for (const cv::Rect& box : boxes) {
        cv::rectangle(roi_mask, box, cv::Scalar(255), cv::FILLED);
        result.boxes.push_back(cv::Rect(box.x + roi_rect.x, box.y + roi_rect.y, box.width, box.height));
    }

    cv::Mat final_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::dilate(roi_mask, roi_mask, final_kernel, cv::Point(-1, -1), 1);
    roi_mask.copyTo(result.mask(roi_rect));
    result.masked_pixels = cv::countNonZero(result.mask);
    return result;
}

}  // namespace remove_subtitle
