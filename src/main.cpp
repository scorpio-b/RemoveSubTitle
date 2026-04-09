#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace fs = std::filesystem;

struct Options {
    std::string input_path;
    std::string output_path;
    int x = 180;
    int y = 820;
    int width = 360;
    int height = 220;
    int temporal_window = 12;
    int patch_radius = 2;
};

void PrintUsage() {
    std::cout
        << "Usage: remove_subtitle <input_video> <output_video> "
        << "[x y width height temporal_window patch_radius]\n";
}

Options ParseArgs(int argc, char** argv) {
    if (argc != 3 && argc != 9) {
        PrintUsage();
        throw std::runtime_error("invalid arguments");
    }

    Options options;
    options.input_path = argv[1];
    options.output_path = argv[2];

    if (argc == 9) {
        options.x = std::stoi(argv[3]);
        options.y = std::stoi(argv[4]);
        options.width = std::stoi(argv[5]);
        options.height = std::stoi(argv[6]);
        options.temporal_window = std::stoi(argv[7]);
        options.patch_radius = std::stoi(argv[8]);
    }

    return options;
}

cv::Rect ClampRect(const cv::Rect& rect, const cv::Size& frame_size) {
    return rect & cv::Rect(0, 0, frame_size.width, frame_size.height);
}

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
    cv::remap(
        reference_bgr,
        warped,
        map_x,
        map_y,
        cv::INTER_LINEAR,
        cv::BORDER_REPLICATE
    );
    return warped;
}

cv::Mat DetectSubtitleMask(const cv::Mat& frame, const cv::Rect& roi_rect) {
    cv::Mat mask(frame.size(), CV_8UC1, cv::Scalar(0));

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

    cv::Mat roi_mask(roi.size(), CV_8UC1, cv::Scalar(0));
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

        const cv::Rect component_rect(
            std::max(0, left - 6),
            std::max(0, top - 6),
            std::min(roi.cols - std::max(0, left - 6), width + 12),
            std::min(roi.rows - std::max(0, top - 6), height + 12)
        );
        cv::rectangle(roi_mask, component_rect, cv::Scalar(255), cv::FILLED);
    }

    cv::Mat final_kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    cv::dilate(roi_mask, roi_mask, final_kernel, cv::Point(-1, -1), 1);

    roi_mask.copyTo(mask(roi_rect));
    return mask;
}

cv::Vec3b SpatialFallbackSample(
    const cv::Mat& frame,
    const cv::Mat& mask,
    int y,
    int x
) {
    std::vector<cv::Vec3b> samples;
    for (int radius = 1; radius <= 12 && samples.size() < 24; ++radius) {
        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                if (std::abs(dx) != radius && std::abs(dy) != radius) {
                    continue;
                }
                const int yy = y + dy;
                const int xx = x + dx;
                if (yy < 0 || yy >= frame.rows || xx < 0 || xx >= frame.cols) {
                    continue;
                }
                if (mask.at<uchar>(yy, xx) == 0) {
                    samples.push_back(frame.at<cv::Vec3b>(yy, xx));
                }
            }
        }
    }

    if (samples.empty()) {
        return frame.at<cv::Vec3b>(y, x);
    }

    cv::Vec3d sum(0.0, 0.0, 0.0);
    for (const cv::Vec3b& sample : samples) {
        sum[0] += sample[0];
        sum[1] += sample[1];
        sum[2] += sample[2];
    }

    const double scale = 1.0 / static_cast<double>(samples.size());
    return cv::Vec3b(
        cv::saturate_cast<uchar>(sum[0] * scale),
        cv::saturate_cast<uchar>(sum[1] * scale),
        cv::saturate_cast<uchar>(sum[2] * scale)
    );
}

cv::Mat RestoreFrame(
    const std::vector<cv::Mat>& frames,
    const std::vector<cv::Mat>& masks,
    int index,
    int temporal_window,
    int patch_radius
) {
    cv::Mat restored = frames[index].clone();
    const cv::Mat& current_mask = masks[index];

    std::vector<cv::Point> masked_points;
    cv::findNonZero(current_mask, masked_points);
    if (masked_points.empty()) {
        return restored;
    }

    struct AlignedReference {
        int frame_index;
        int offset;
        cv::Mat warped_frame;
    };

    std::vector<AlignedReference> aligned_references;
    aligned_references.reserve(temporal_window * 2);
    for (int offset = 1; offset <= temporal_window; ++offset) {
        const int prev = index - offset;
        if (prev >= 0) {
            aligned_references.push_back({
                prev,
                offset,
                WarpReferenceToCurrent(frames[index], frames[prev])
            });
        }

        const int next = index + offset;
        if (next < static_cast<int>(frames.size())) {
            aligned_references.push_back({
                next,
                offset,
                WarpReferenceToCurrent(frames[index], frames[next])
            });
        }
    }

    for (const cv::Point& pt : masked_points) {
        struct Candidate {
            double score;
            cv::Vec3b pixel;
        };
        std::vector<Candidate> candidates;

        for (const AlignedReference& reference : aligned_references) {
            if (masks[reference.frame_index].at<uchar>(pt) != 0) {
                continue;
            }

            const double score = PatchDifference(
                frames[index],
                reference.warped_frame,
                pt,
                patch_radius
            ) + reference.offset * 0.35;

            if (score > 28.0) {
                continue;
            }

            candidates.push_back({score, reference.warped_frame.at<cv::Vec3b>(pt)});
        }

        if (!candidates.empty()) {
            std::sort(
                candidates.begin(),
                candidates.end(),
                [](const Candidate& lhs, const Candidate& rhs) {
                    return lhs.score < rhs.score;
                }
            );

            const int use_count = std::min<int>(3, candidates.size());
            cv::Vec3d sum(0.0, 0.0, 0.0);
            double weight_sum = 0.0;
            for (int i = 0; i < use_count; ++i) {
                const double weight = 1.0 / std::max(1.0, candidates[i].score);
                sum[0] += static_cast<double>(candidates[i].pixel[0]) * weight;
                sum[1] += static_cast<double>(candidates[i].pixel[1]) * weight;
                sum[2] += static_cast<double>(candidates[i].pixel[2]) * weight;
                weight_sum += weight;
            }

            restored.at<cv::Vec3b>(pt) = cv::Vec3b(
                cv::saturate_cast<uchar>(sum[0] / weight_sum),
                cv::saturate_cast<uchar>(sum[1] / weight_sum),
                cv::saturate_cast<uchar>(sum[2] / weight_sum)
            );
            continue;
        }

        restored.at<cv::Vec3b>(pt) = SpatialFallbackSample(restored, current_mask, pt.y, pt.x);
    }

    return restored;
}

int main(int argc, char** argv) {
    try {
        const Options options = ParseArgs(argc, argv);
        if (!fs::exists(options.input_path)) {
            throw std::runtime_error("input video does not exist");
        }

        cv::VideoCapture capture(options.input_path);
        if (!capture.isOpened()) {
            throw std::runtime_error("failed to open input video");
        }

        const int frame_width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
        const int frame_height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
        double fps = capture.get(cv::CAP_PROP_FPS);
        if (fps <= 0.0) {
            fps = 24.0;
        }

        const cv::Size frame_size(frame_width, frame_height);
        const cv::Rect roi_rect = ClampRect(
            cv::Rect(options.x, options.y, options.width, options.height),
            frame_size
        );
        if (roi_rect.empty()) {
            throw std::runtime_error("subtitle roi is outside the frame");
        }

        std::vector<cv::Mat> frames;
        for (cv::Mat frame; capture.read(frame); ) {
            frames.push_back(frame.clone());
        }
        if (frames.empty()) {
            throw std::runtime_error("no frames read from input video");
        }

        std::vector<cv::Mat> masks;
        masks.reserve(frames.size());
        for (const cv::Mat& frame : frames) {
            masks.push_back(DetectSubtitleMask(frame, roi_rect));
        }

        const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        cv::VideoWriter writer(options.output_path, fourcc, fps, frame_size, true);
        if (!writer.isOpened()) {
            throw std::runtime_error("failed to open output video");
        }

        int subtitle_frames = 0;
        for (int index = 0; index < static_cast<int>(frames.size()); ++index) {
            const bool has_subtitle = cv::countNonZero(masks[index]) > 0;
            subtitle_frames += has_subtitle ? 1 : 0;
            const cv::Mat restored = has_subtitle
                ? RestoreFrame(frames, masks, index, options.temporal_window, options.patch_radius)
                : frames[index];
            writer.write(restored);
        }

        std::cout << "Processed video written to: " << options.output_path << '\n';
        std::cout << "Subtitle ROI: "
                  << roi_rect.x << "," << roi_rect.y << ","
                  << roi_rect.width << "," << roi_rect.height << '\n';
        std::cout << "Frames with subtitle mask: " << subtitle_frames << "/" << frames.size() << '\n';
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
}
