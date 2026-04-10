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

cv::Mat ExtractContourMask(
    const cv::Mat& solid_mask,
    int min_area,
    int thickness
) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(solid_mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat contour_mask(solid_mask.size(), CV_8UC1, cv::Scalar(0));
    for (const auto& contour : contours) {
        const double area = cv::contourArea(contour);
        if (area < min_area) {
            continue;
        }
        cv::drawContours(
            contour_mask,
            std::vector<std::vector<cv::Point>>{contour},
            -1,
            cv::Scalar(255),
            std::max(1, thickness)
        );
    }
    return contour_mask;
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

cv::Mat FilterCharacterIslands(const cv::Mat& mask) {
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int component_count =
        cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8);

    cv::Mat filtered(mask.size(), CV_8UC1, cv::Scalar(0));
    for (int label = 1; label < component_count; ++label) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        const int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
        const int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);

        if (area < 10) {
            continue;
        }
        if (height < 8 || height > 80) {
            continue;
        }
        if (width < 2 || width > 48) {
            continue;
        }

        const double aspect_ratio = static_cast<double>(width) / static_cast<double>(std::max(1, height));
        if (aspect_ratio > 1.35) {
            continue;
        }

        cv::Mat component_mask = labels == label;
        filtered.setTo(255, component_mask);
    }
    return filtered;
}

cv::Mat MergeStrokeClusters(const cv::Mat& strokes) {
    cv::Mat merged = strokes.clone();
    cv::Mat merge_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
    cv::morphologyEx(merged, merged, cv::MORPH_CLOSE, merge_kernel);
    merged = FillClosedRegions(merged);

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int component_count =
        cv::connectedComponentsWithStats(merged, labels, stats, centroids, 8);

    cv::Mat filtered(merged.size(), CV_8UC1, cv::Scalar(0));
    for (int label = 1; label < component_count; ++label) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        const int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
        const int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);

        if (area < 12 || area > 900) {
            continue;
        }
        if (width < 3 || height < 8 || width > 42 || height > 64) {
            continue;
        }

        cv::Mat component_mask = labels == label;
        filtered.setTo(255, component_mask);
    }

    return filtered;
}

cv::Mat FilterStrokeLikeComponents(const cv::Mat& mask) {
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int component_count =
        cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8);

    cv::Mat filtered(mask.size(), CV_8UC1, cv::Scalar(0));
    for (int label = 1; label < component_count; ++label) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        const int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
        const int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);

        if (area < 3 || area > 320) {
            continue;
        }
        if (width < 1 || height < 3 || width > 36 || height > 52) {
            continue;
        }

        const int major = std::max(width, height);
        const int minor = std::max(1, std::min(width, height));
        const double elongation = static_cast<double>(major) / static_cast<double>(minor);
        if (elongation < 1.15) {
            continue;
        }

        cv::Mat component_mask = labels == label;
        filtered.setTo(255, component_mask);
    }

    return filtered;
}

cv::Mat ValidateByDarkRing(
    const cv::Mat& white_core,
    const cv::Mat& dark_outline
) {
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int component_count =
        cv::connectedComponentsWithStats(white_core, labels, stats, centroids, 8);

    cv::Mat filtered(white_core.size(), CV_8UC1, cv::Scalar(0));
    cv::Mat ring_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));

    for (int label = 1; label < component_count; ++label) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        const int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
        const int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
        if (area < 6 || width < 2 || height < 6 || width > 56 || height > 72) {
            continue;
        }

        cv::Mat component_mask = labels == label;
        component_mask.convertTo(component_mask, CV_8UC1, 255);

        cv::Mat outer_ring;
        cv::dilate(component_mask, outer_ring, ring_kernel, cv::Point(-1, -1), 1);
        cv::subtract(outer_ring, component_mask, outer_ring);

        const int ring_pixels = cv::countNonZero(outer_ring);
        if (ring_pixels == 0) {
            continue;
        }

        cv::Mat dark_on_ring;
        cv::bitwise_and(outer_ring, dark_outline, dark_on_ring);
        const double dark_ratio =
            static_cast<double>(cv::countNonZero(dark_on_ring)) / static_cast<double>(ring_pixels);
        if (dark_ratio < 0.10) {
            continue;
        }

        filtered.setTo(255, component_mask);
    }

    return filtered;
}

cv::Mat ExtractClosedStrokeSeed(const cv::Mat& white_candidates) {
    cv::Mat contours_source = white_candidates.clone();
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(
        contours_source,
        contours,
        hierarchy,
        cv::RETR_CCOMP,
        cv::CHAIN_APPROX_SIMPLE
    );

    cv::Mat closed_seed(white_candidates.size(), CV_8UC1, cv::Scalar(0));
    for (int index = 0; index < static_cast<int>(contours.size()); ++index) {
        const double area = cv::contourArea(contours[index]);
        if (area < 6.0 || area > 1200.0) {
            continue;
        }

        const cv::Rect box = cv::boundingRect(contours[index]);
        if (box.height < 6 || box.height > 56 || box.width < 2 || box.width > 72) {
            continue;
        }

        cv::drawContours(
            closed_seed,
            contours,
            index,
            cv::Scalar(255),
            cv::FILLED,
            cv::LINE_8,
            hierarchy
        );
    }

    return closed_seed;
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

void ExtractCharacterMask(
    const cv::Mat& gray,
    const cv::Mat& low_saturation,
    const cv::Mat& bright_value,
    const cv::Mat& dark_value,
    const cv::Rect& band_box,
    int repair_expand,
    cv::Mat& outline_mask_out,
    cv::Mat& repair_mask_out
) {
    outline_mask_out = cv::Mat(gray.size(), CV_8UC1, cv::Scalar(0));
    repair_mask_out = cv::Mat(gray.size(), CV_8UC1, cv::Scalar(0));
    if (band_box.empty()) {
        return;
    }

    const cv::Rect clipped = band_box & cv::Rect(0, 0, gray.cols, gray.rows);
    if (clipped.empty()) {
        return;
    }

    cv::Mat band_gray = gray(clipped);
    cv::Mat band_sat = low_saturation(clipped);
    cv::Mat band_bright = bright_value(clipped);
    cv::Mat band_dark = dark_value(clipped);

    cv::Mat local_background;
    cv::GaussianBlur(band_gray, local_background, cv::Size(0, 0), 5.0);

    cv::Mat local_contrast16;
    cv::subtract(band_gray, local_background, local_contrast16, cv::noArray(), CV_16S);

    cv::Mat local_contrast;
    cv::convertScaleAbs(local_contrast16, local_contrast);

    cv::Mat strong_white;
    cv::inRange(band_gray, 150, 255, strong_white);
    cv::bitwise_and(strong_white, band_sat, strong_white);

    cv::Mat soft_white;
    cv::inRange(band_gray, 132, 255, soft_white);
    cv::bitwise_and(soft_white, band_sat, soft_white);

    cv::Mat weak_white;
    cv::inRange(band_gray, 110, 235, weak_white);
    cv::bitwise_and(weak_white, band_sat, weak_white);

    cv::Mat contrast_white;
    cv::threshold(local_contrast, contrast_white, 12, 255, cv::THRESH_BINARY);
    cv::bitwise_and(contrast_white, band_sat, contrast_white);

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
    cv::morphologyEx(soft_white, soft_white, cv::MORPH_CLOSE, stroke_kernel);
    cv::morphologyEx(weak_white, weak_white, cv::MORPH_OPEN, stroke_kernel);

    cv::Mat white_core = strong_white.clone();
    cv::bitwise_or(white_core, soft_white, white_core);
    white_core = RemoveSmallComponents(white_core, 3);
    white_core = FilterStrokeLikeComponents(white_core);
    white_core = ValidateByDarkRing(white_core, dark_outline);
    if (cv::countNonZero(white_core) == 0) {
        return;
    }

    cv::Mat stroke_candidate = soft_white.clone();
    cv::bitwise_or(stroke_candidate, strong_white, stroke_candidate);
    cv::Mat weak_candidate;
    cv::bitwise_and(weak_white, contrast_white, weak_candidate);
    cv::bitwise_or(stroke_candidate, weak_candidate, stroke_candidate);
    stroke_candidate = RemoveSmallComponents(stroke_candidate, 3);
    cv::Mat grown_white = GrowFromSeed(white_core, stroke_candidate, 3);
    cv::Mat candidate_islands = MergeStrokeClusters(grown_white);
    candidate_islands = FilterCharacterIslands(candidate_islands);
    if (cv::countNonZero(candidate_islands) == 0) {
        return;
    }

    cv::Mat contour_mask = ExtractContourMask(candidate_islands, 10, 2);
    if (cv::countNonZero(contour_mask) == 0) {
        return;
    }

    cv::Mat repair_fill = FillClosedRegions(contour_mask);
    cv::Mat repair_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    if (repair_expand > 0) {
        const int kernel_size = std::max(3, repair_expand | 1);
        cv::Mat expand_kernel = cv::getStructuringElement(
            cv::MORPH_ELLIPSE,
            cv::Size(kernel_size, kernel_size)
        );
        const int iterations = 1;
        cv::dilate(repair_fill, repair_fill, expand_kernel, cv::Point(-1, -1), iterations);
        cv::morphologyEx(repair_fill, repair_fill, cv::MORPH_CLOSE, expand_kernel);
    }
    repair_fill = FillClosedRegions(repair_fill);
    cv::morphologyEx(repair_fill, repair_fill, cv::MORPH_CLOSE, repair_kernel);
    repair_fill.copyTo(repair_mask_out(clipped));

    contour_mask.copyTo(outline_mask_out(clipped));
}

}  // namespace

DetectionResult DetectSubtitleMask(
    const cv::Mat& frame,
    const cv::Rect& roi_rect,
    int repair_expand
) {
    DetectionResult result;
    result.mask = cv::Mat(frame.size(), CV_8UC1, cv::Scalar(0));
    result.repair_mask = cv::Mat(frame.size(), CV_8UC1, cv::Scalar(0));
    result.expanded_outline_mask = cv::Mat(frame.size(), CV_8UC1, cv::Scalar(0));
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

    cv::Mat local_outline_mask;
    cv::Mat local_repair_mask;
    ExtractCharacterMask(
        gray,
        low_saturation,
        bright_value,
        dark_value,
        band_box,
        repair_expand,
        local_outline_mask,
        local_repair_mask
    );
    local_outline_mask.copyTo(result.mask(roi_rect));
    local_repair_mask.copyTo(result.repair_mask(roi_rect));
    cv::Mat expanded_outline = ExtractContourMask(local_repair_mask, 10, 2);
    expanded_outline.copyTo(result.expanded_outline_mask(roi_rect));
    result.masked_pixels = cv::countNonZero(result.mask);
    return result;
}

}  // namespace remove_subtitle
