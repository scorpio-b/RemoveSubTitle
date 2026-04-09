#include "remove_subtitle/subtitle_detector.hpp"

#include <algorithm>

namespace remove_subtitle {

namespace {

cv::Rect ExtractSubtitleBand(const cv::Mat& band_mask) {
    const int rows = band_mask.rows;
    const int cols = band_mask.cols;

    int best_start = -1;
    int best_end = -1;
    int current_start = -1;
    int best_score = 0;
    int current_score = 0;

    for (int y = static_cast<int>(rows * 0.45); y < rows; ++y) {
        const int row_score = cv::countNonZero(band_mask.row(y));
        const bool active = row_score >= cols / 14;

        if (active) {
            if (current_start < 0) {
                current_start = y;
                current_score = 0;
            }
            current_score += row_score;
        } else if (current_start >= 0) {
            const int current_end = y - 1;
            if (current_score > best_score) {
                best_score = current_score;
                best_start = current_start;
                best_end = current_end;
            }
            current_start = -1;
            current_score = 0;
        }
    }

    if (current_start >= 0) {
        const int current_end = rows - 1;
        if (current_score > best_score) {
            best_score = current_score;
            best_start = current_start;
            best_end = current_end;
        }
    }

    if (best_start < 0 || best_end < 0) {
        return {};
    }

    cv::Mat slice = band_mask.rowRange(best_start, best_end + 1);
    std::vector<cv::Point> points;
    cv::findNonZero(slice, points);
    if (points.empty()) {
        return {};
    }

    cv::Rect bounds = cv::boundingRect(points);
    bounds.y += best_start;

    const int pad_x = 10;
    const int pad_y = 8;
    const int x = std::max(0, bounds.x - pad_x);
    const int y = std::max(0, bounds.y - pad_y);
    const int width = std::min(cols - x, bounds.width + pad_x * 2);
    const int height = std::min(rows - y, bounds.height + pad_y * 2);
    return cv::Rect(x, y, width, height);
}

cv::Mat ExtractFineTextMask(
    const cv::Mat& gray,
    const cv::Mat& low_saturation,
    const cv::Mat& bright_value,
    const cv::Mat& dark_outline,
    const cv::Rect& band_box
) {
    cv::Mat fine_mask(gray.size(), CV_8UC1, cv::Scalar(0));
    if (band_box.empty()) {
        return fine_mask;
    }

    const cv::Rect clipped = band_box & cv::Rect(0, 0, gray.cols, gray.rows);
    const int focus_y = clipped.y + clipped.height / 10;
    const int focus_height = std::max(10, clipped.height * 9 / 20);
    const cv::Rect focus_rect(
        clipped.x,
        focus_y,
        clipped.width,
        std::min(gray.rows - focus_y, focus_height)
    );

    cv::Mat band_gray = gray(focus_rect);
    cv::Mat band_sat = low_saturation(focus_rect);
    cv::Mat band_bright = bright_value(focus_rect);
    cv::Mat band_dark = dark_outline(focus_rect);

    cv::Mat local_binary;
    cv::adaptiveThreshold(
        band_gray,
        local_binary,
        255,
        cv::ADAPTIVE_THRESH_GAUSSIAN_C,
        cv::THRESH_BINARY,
        21,
        -4
    );

    cv::Mat bright_text;
    cv::bitwise_and(local_binary, band_sat, bright_text);
    cv::bitwise_and(bright_text, band_bright, bright_text);

    cv::Mat dark_text;
    cv::bitwise_and(local_binary, band_dark, dark_text);

    cv::Mat text_mask;
    cv::bitwise_or(bright_text, dark_text, text_mask);

    cv::Mat open_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(text_mask, text_mask, cv::MORPH_OPEN, open_kernel);

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int component_count = cv::connectedComponentsWithStats(text_mask, labels, stats, centroids, 8);

    cv::Mat filtered_mask(text_mask.size(), CV_8UC1, cv::Scalar(0));
    struct AcceptedComponent {
        int label;
        cv::Rect box;
    };
    std::vector<AcceptedComponent> accepted_components;
    for (int label = 1; label < component_count; ++label) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        const int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
        const int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
        const int left = stats.at<int>(label, cv::CC_STAT_LEFT);
        const int top = stats.at<int>(label, cv::CC_STAT_TOP);

        if (area < 8 || area > 450) {
            continue;
        }

        if (height < 6 || height > focus_rect.height - 2) {
            continue;
        }

        if (width < 2 || width > 42) {
            continue;
        }

        const double aspect_ratio = static_cast<double>(width) / static_cast<double>(height);
        if (aspect_ratio > 4.5) {
            continue;
        }

        accepted_components.push_back({label, cv::Rect(left, top, width, height)});
    }

    if (accepted_components.empty()) {
        return fine_mask;
    }

    std::sort(
        accepted_components.begin(),
        accepted_components.end(),
        [](const AcceptedComponent& lhs, const AcceptedComponent& rhs) {
            return lhs.box.x < rhs.box.x;
        }
    );

    int median_center_y = 0;
    std::vector<int> centers;
    centers.reserve(accepted_components.size());
    for (const AcceptedComponent& component : accepted_components) {
        centers.push_back(component.box.y + component.box.height / 2);
    }
    std::sort(centers.begin(), centers.end());
    median_center_y = centers[centers.size() / 2];

    std::vector<int> heights;
    heights.reserve(accepted_components.size());
    for (const AcceptedComponent& component : accepted_components) {
        heights.push_back(component.box.height);
    }
    std::sort(heights.begin(), heights.end());
    const int median_height = heights[heights.size() / 2];

    std::vector<AcceptedComponent> grouped_components;
    grouped_components.reserve(accepted_components.size());
    int previous_right = -1;
    for (const AcceptedComponent& component : accepted_components) {
        const int center_y = component.box.y + component.box.height / 2;
        if (std::abs(center_y - median_center_y) > 8) {
            continue;
        }

        if (std::abs(component.box.height - median_height) > 6) {
            continue;
        }

        if (previous_right >= 0) {
            const int gap = component.box.x - previous_right;
            if (gap > median_height * 2) {
                continue;
            }
        }

        grouped_components.push_back(component);
        previous_right = component.box.x + component.box.width;
    }

    if (grouped_components.size() < 2) {
        return fine_mask;
    }

    for (const AcceptedComponent& component : grouped_components) {
        cv::Mat component_mask = labels == component.label;
        filtered_mask.setTo(255, component_mask);
    }

    cv::Mat dilate_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(filtered_mask, filtered_mask, dilate_kernel, cv::Point(-1, -1), 1);

    filtered_mask.copyTo(fine_mask(focus_rect));
    return fine_mask;
}

}  // namespace

DetectionResult DetectSubtitleMask(const cv::Mat& frame, const cv::Rect& roi_rect) {
    DetectionResult result;
    result.mask = cv::Mat(frame.size(), CV_8UC1, cv::Scalar(0));
    result.band_mask = cv::Mat(frame.size(), CV_8UC1, cv::Scalar(0));

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

    cv::Mat band_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(31, 3));
    cv::Mat band_mask;
    cv::morphologyEx(dilated, band_mask, cv::MORPH_CLOSE, band_kernel);

    std::vector<cv::Rect> boxes;
    const cv::Rect band_box = ExtractSubtitleBand(band_mask);
    if (!band_box.empty()) {
        boxes.push_back(band_box);
    }

    for (const cv::Rect& box : boxes) {
        cv::rectangle(result.band_mask(roi_rect), box, cv::Scalar(255), cv::FILLED);
        result.boxes.push_back(cv::Rect(box.x + roi_rect.x, box.y + roi_rect.y, box.width, box.height));
    }

    cv::Mat fine_mask = cv::Mat(roi.size(), CV_8UC1, cv::Scalar(0));
    if (!band_box.empty()) {
        fine_mask = ExtractFineTextMask(
            gray,
            low_saturation,
            bright_value,
            dark_outline,
            band_box
        );
    }

    cv::Mat final_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    cv::dilate(fine_mask, fine_mask, final_kernel, cv::Point(-1, -1), 1);
    fine_mask.copyTo(result.mask(roi_rect));
    result.masked_pixels = cv::countNonZero(result.mask);
    return result;
}

}  // namespace remove_subtitle
