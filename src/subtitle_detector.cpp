#include "remove_subtitle/subtitle_detector.hpp"

#include <algorithm>

namespace remove_subtitle {

namespace {

cv::Mat FillClosedRegions(const cv::Mat& binary_mask) {
    cv::Mat filled = binary_mask.clone();
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(filled.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    filled.setTo(0);
    for (const auto& contour : contours) {
        const double area = cv::contourArea(contour);
        if (area < 8.0) {
            continue;
        }
        cv::drawContours(filled, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(255), cv::FILLED);
    }
    return filled;
}

cv::Mat ExtractThinOutline(const cv::Mat& solid_mask) {
    cv::Mat eroded;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::erode(solid_mask, eroded, kernel, cv::Point(-1, -1), 1);

    cv::Mat outline;
    cv::subtract(solid_mask, eroded, outline);
    return outline;
}

cv::Mat GrowFromSeed(
    const cv::Mat& seed_mask,
    const cv::Mat& candidate_mask,
    int iterations
) {
    cv::Mat grown = seed_mask.clone();
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    for (int i = 0; i < iterations; ++i) {
        cv::Mat expanded;
        cv::dilate(grown, expanded, kernel, cv::Point(-1, -1), 1);
        cv::bitwise_and(expanded, candidate_mask, expanded);
        cv::bitwise_or(grown, expanded, grown);
    }
    return grown;
}

cv::Mat RemoveSmallComponents(const cv::Mat& mask, int min_area) {
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int component_count =
        cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8);

    cv::Mat filtered(mask.size(), CV_8UC1, cv::Scalar(0));
    for (int label = 1; label < component_count; ++label) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area < min_area) {
            continue;
        }
        cv::Mat component_mask = labels == label;
        filtered.setTo(255, component_mask);
    }
    return filtered;
}

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

    cv::Mat strong_white;
    cv::inRange(band_gray, 150, 255, strong_white);
    cv::bitwise_and(strong_white, band_sat, strong_white);

    cv::Mat soft_white;
    cv::inRange(band_gray, 120, 255, soft_white);
    cv::bitwise_and(soft_white, band_sat, soft_white);

    cv::Mat gray_white;
    cv::inRange(band_gray, 105, 210, gray_white);
    cv::bitwise_and(gray_white, band_sat, gray_white);

    cv::Mat dark_outline;
    cv::adaptiveThreshold(
        band_gray,
        dark_outline,
        255,
        cv::ADAPTIVE_THRESH_GAUSSIAN_C,
        cv::THRESH_BINARY_INV,
        15,
        4
    );
    cv::bitwise_and(dark_outline, band_dark, dark_outline);

    cv::Mat stroke_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(strong_white, strong_white, cv::MORPH_CLOSE, stroke_kernel);

    cv::Mat outline_neighbor;
    cv::dilate(dark_outline, outline_neighbor, stroke_kernel, cv::Point(-1, -1), 1);

    cv::Mat edge_seed;
    cv::bitwise_and(gray_white, outline_neighbor, edge_seed);
    cv::morphologyEx(edge_seed, edge_seed, cv::MORPH_OPEN, stroke_kernel);

    cv::Mat soft_candidate;
    cv::bitwise_and(soft_white, outline_neighbor, soft_candidate);
    cv::bitwise_or(soft_candidate, edge_seed, soft_candidate);
    cv::bitwise_or(soft_candidate, strong_white, soft_candidate);

    cv::Mat combined_seed;
    cv::bitwise_or(strong_white, edge_seed, combined_seed);
    cv::Mat grown_white = GrowFromSeed(combined_seed, soft_candidate, 4);
    cv::Mat filled_white = FillClosedRegions(grown_white);
    cv::Mat bridge_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(filled_white, filled_white, cv::MORPH_CLOSE, bridge_kernel);

    cv::Mat sentence_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 3));
    cv::Mat sentence_mask;
    cv::morphologyEx(filled_white, sentence_mask, cv::MORPH_CLOSE, sentence_kernel);
    sentence_mask = RemoveSmallComponents(sentence_mask, 24);
    if (cv::countNonZero(sentence_mask) == 0) {
        return result;
    }

    cv::Mat filtered_fill;
    cv::bitwise_and(filled_white, sentence_mask, filtered_fill);
    filtered_fill = RemoveSmallComponents(filtered_fill, 10);
    cv::morphologyEx(filtered_fill, filtered_fill, cv::MORPH_CLOSE, bridge_kernel);
    cv::Mat thin_outline = ExtractThinOutline(filtered_fill);
    thin_outline.copyTo(result(clipped));
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
