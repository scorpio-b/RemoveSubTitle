#include "remove_subtitle/subtitle_detector.hpp"

#include <algorithm>
#include <cctype>
#include <memory>
#include <mutex>

#include <tesseract/baseapi.h>

namespace remove_subtitle {

namespace {

class OcrLineFilter {
public:
    OcrLineFilter() {
        if (api_.Init(nullptr, "eng")) {
            enabled_ = false;
            return;
        }
        api_.SetPageSegMode(tesseract::PSM_SINGLE_LINE);
        enabled_ = true;
    }

    double TextLikelihood(const cv::Mat& roi) {
        if (!enabled_ || roi.empty()) {
            return 0.0;
        }

        cv::Mat gray;
        if (roi.channels() == 3) {
            cv::cvtColor(roi, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = roi;
        }

        cv::Mat scaled;
        cv::resize(gray, scaled, cv::Size(), 2.0, 2.0, cv::INTER_CUBIC);
        cv::Mat binary;
        cv::threshold(scaled, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

        std::lock_guard<std::mutex> lock(mutex_);
        api_.SetImage(binary.data, binary.cols, binary.rows, 1, static_cast<int>(binary.step));
        api_.Recognize(nullptr);
        const char* text = api_.GetUTF8Text();
        const int confidence = api_.MeanTextConf();

        double score = confidence >= 15 ? 1.0 : 0.0;
        if (text != nullptr) {
            std::string raw_text(text);
            delete[] text;
            const auto it = std::find_if(raw_text.begin(), raw_text.end(), [](unsigned char c) {
                return !std::isspace(c);
            });
            if (it != raw_text.end()) {
                score += 1.0;
            }
        }

        api_.Clear();
        return score;
    }

private:
    tesseract::TessBaseAPI api_;
    std::mutex mutex_;
    bool enabled_ = false;
};

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

    cv::Mat band_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(31, 3));
    cv::Mat band_mask;
    cv::morphologyEx(dilated, band_mask, cv::MORPH_CLOSE, band_kernel);

    std::vector<cv::Rect> boxes;
    const cv::Rect band_box = ExtractSubtitleBand(band_mask);
    if (!band_box.empty()) {
        boxes.push_back(band_box);
    }

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
