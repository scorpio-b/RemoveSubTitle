#include "remove_subtitle/subtitle_detector.hpp"

#include <algorithm>

namespace remove_subtitle {

namespace {

cv::Rect ExtractSubtitleBand(const cv::Mat& candidate_mask) {
    const int rows = candidate_mask.rows;
    const int cols = candidate_mask.cols;

    int best_start = -1;
    int best_end = -1;
    int current_start = -1;
    int best_score = 0;
    int current_score = 0;

    for (int y = rows / 3; y < rows; ++y) {
        const int row_score = cv::countNonZero(candidate_mask.row(y));
        const bool active = row_score >= std::max(18, cols / 18);

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

    cv::Mat band_slice = candidate_mask.rowRange(best_start, best_end + 1);
    std::vector<cv::Point> points;
    cv::findNonZero(band_slice, points);
    if (points.empty()) {
        return {};
    }

    cv::Rect bounds = cv::boundingRect(points);
    bounds.y += best_start;

    const int pad_x = 8;
    const int pad_y = 6;
    const int x = std::max(0, bounds.x - pad_x);
    const int y = std::max(0, bounds.y - pad_y);
    const int width = std::min(cols - x, bounds.width + pad_x * 2);
    const int height = std::min(rows - y, bounds.height + pad_y * 2);
    return cv::Rect(x, y, width, height);
}

cv::Mat ExtractCharacterMask(
    const cv::Mat& gray,
    const cv::Mat& low_saturation,
    const cv::Mat& bright_value,
    const cv::Mat& dark_value,
    const cv::Rect& band_box
) {
    cv::Mat result(gray.size(), CV_8UC1, cv::Scalar(0));
    if (band_box.empty()) {
        return result;
    }

    const cv::Rect clipped = band_box & cv::Rect(0, 0, gray.cols, gray.rows);
    if (clipped.empty()) {
        return result;
    }

    cv::Mat band_gray = gray(clipped);
    cv::Mat band_sat = low_saturation(clipped);
    cv::Mat band_bright = bright_value(clipped);
    cv::Mat band_dark = dark_value(clipped);

    cv::Mat core_mask;
    cv::threshold(band_gray, core_mask, 170, 255, cv::THRESH_BINARY);
    cv::bitwise_and(core_mask, band_sat, core_mask);
    cv::bitwise_and(core_mask, band_bright, core_mask);

    cv::Mat outline_mask;
    cv::adaptiveThreshold(
        band_gray,
        outline_mask,
        255,
        cv::ADAPTIVE_THRESH_GAUSSIAN_C,
        cv::THRESH_BINARY_INV,
        17,
        6
    );
    cv::bitwise_and(outline_mask, band_dark, outline_mask);

    cv::Mat expanded_core;
    cv::Mat core_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::dilate(core_mask, expanded_core, core_kernel, cv::Point(-1, -1), 1);

    cv::Mat validated_text;
    cv::bitwise_and(expanded_core, outline_mask, validated_text);
    cv::bitwise_or(validated_text, core_mask, validated_text);

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int component_count =
        cv::connectedComponentsWithStats(validated_text, labels, stats, centroids, 8);

    struct Glyph {
        int label;
        cv::Rect box;
    };
    std::vector<Glyph> glyphs;
    for (int label = 1; label < component_count; ++label) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        const int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
        const int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
        const int left = stats.at<int>(label, cv::CC_STAT_LEFT);
        const int top = stats.at<int>(label, cv::CC_STAT_TOP);

        if (area < 12 || area > 320) {
            continue;
        }
        if (height < 10 || height > 34) {
            continue;
        }
        if (width < 6 || width > 42) {
            continue;
        }

        const double aspect_ratio = static_cast<double>(width) / static_cast<double>(height);
        if (aspect_ratio < 0.18 || aspect_ratio > 2.8) {
            continue;
        }

        glyphs.push_back({label, cv::Rect(left, top, width, height)});
    }

    if (glyphs.empty()) {
        return result;
    }

    std::sort(
        glyphs.begin(),
        glyphs.end(),
        [](const Glyph& lhs, const Glyph& rhs) { return lhs.box.x < rhs.box.x; }
    );

    std::vector<int> centers_y;
    std::vector<int> heights;
    centers_y.reserve(glyphs.size());
    heights.reserve(glyphs.size());
    for (const Glyph& glyph : glyphs) {
        centers_y.push_back(glyph.box.y + glyph.box.height / 2);
        heights.push_back(glyph.box.height);
    }
    std::sort(centers_y.begin(), centers_y.end());
    std::sort(heights.begin(), heights.end());
    const int median_center_y = centers_y[centers_y.size() / 2];
    const int median_height = heights[heights.size() / 2];

    cv::Mat filtered_mask(validated_text.size(), CV_8UC1, cv::Scalar(0));
    int accepted_count = 0;
    int previous_right = -1;
    for (const Glyph& glyph : glyphs) {
        const int center_y = glyph.box.y + glyph.box.height / 2;
        if (std::abs(center_y - median_center_y) > 6) {
            continue;
        }
        if (std::abs(glyph.box.height - median_height) > 5) {
            continue;
        }

        if (previous_right >= 0) {
            const int gap = glyph.box.x - previous_right;
            if (gap > median_height * 2) {
                continue;
            }
        }

        cv::Mat component_mask = labels == glyph.label;
        filtered_mask.setTo(255, component_mask);
        previous_right = glyph.box.x + glyph.box.width;
        ++accepted_count;
    }

    if (accepted_count == 0) {
        return result;
    }

    cv::Mat final_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::dilate(filtered_mask, filtered_mask, final_kernel, cv::Point(-1, -1), 1);
    filtered_mask.copyTo(result(clipped));
    return result;
}

}  // namespace

DetectionResult DetectSubtitleMask(const cv::Mat& frame, const cv::Rect& roi_rect) {
    DetectionResult result;
    result.mask = cv::Mat(frame.size(), CV_8UC1, cv::Scalar(0));
    result.band_mask = cv::Mat(frame.size(), CV_8UC1, cv::Scalar(0));

    cv::Mat roi = frame(roi_rect);
    cv::Mat gray;
    cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);

    cv::Mat hsv;
    cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);
    std::vector<cv::Mat> hsv_channels;
    cv::split(hsv, hsv_channels);

    cv::Mat low_saturation;
    cv::threshold(hsv_channels[1], low_saturation, 105, 255, cv::THRESH_BINARY_INV);

    cv::Mat bright_value;
    cv::threshold(hsv_channels[2], bright_value, 150, 255, cv::THRESH_BINARY);

    cv::Mat dark_value;
    cv::threshold(hsv_channels[2], dark_value, 95, 255, cv::THRESH_BINARY_INV);

    cv::Mat white_core;
    cv::threshold(gray, white_core, 175, 255, cv::THRESH_BINARY);
    cv::bitwise_and(white_core, low_saturation, white_core);
    cv::bitwise_and(white_core, bright_value, white_core);

    cv::Mat outline_candidate;
    cv::adaptiveThreshold(
        gray,
        outline_candidate,
        255,
        cv::ADAPTIVE_THRESH_GAUSSIAN_C,
        cv::THRESH_BINARY_INV,
        17,
        6
    );
    cv::bitwise_and(outline_candidate, dark_value, outline_candidate);

    cv::Mat expanded_core;
    cv::Mat core_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::dilate(white_core, expanded_core, core_kernel, cv::Point(-1, -1), 1);

    cv::Mat candidate_mask;
    cv::bitwise_and(expanded_core, outline_candidate, candidate_mask);
    cv::bitwise_or(candidate_mask, white_core, candidate_mask);

    cv::Mat band_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(21, 3));
    cv::morphologyEx(candidate_mask, candidate_mask, cv::MORPH_CLOSE, band_kernel);

    const cv::Rect band_box = ExtractSubtitleBand(candidate_mask);
    if (!band_box.empty()) {
        cv::rectangle(result.band_mask(roi_rect), band_box, cv::Scalar(255), cv::FILLED);
        result.boxes.push_back(cv::Rect(
            band_box.x + roi_rect.x,
            band_box.y + roi_rect.y,
            band_box.width,
            band_box.height
        ));
    }

    const cv::Mat fine_mask = ExtractCharacterMask(
        gray,
        low_saturation,
        bright_value,
        dark_value,
        band_box
    );

    fine_mask.copyTo(result.mask(roi_rect));
    result.masked_pixels = cv::countNonZero(result.mask);
    return result;
}

}  // namespace remove_subtitle
