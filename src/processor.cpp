#include "remove_subtitle/processor.hpp"

#include "remove_subtitle/geometry.hpp"
#include "remove_subtitle/region_restorer.hpp"
#include "remove_subtitle/subtitle_detector.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;

namespace remove_subtitle {

int ProcessVideo(const Options& options) {
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

    const bool write_debug = !options.debug_dir.empty();
    if (write_debug) {
        fs::create_directories(options.debug_dir);
    }

    std::vector<cv::Mat> frames;
    for (cv::Mat frame; capture.read(frame);) {
        frames.push_back(frame.clone());
    }
    if (frames.empty()) {
        throw std::runtime_error("no frames read from input video");
    }

    std::vector<cv::Mat> masks;
    std::vector<DetectionResult> detections;
    masks.reserve(frames.size());
    detections.reserve(frames.size());
    for (int index = 0; index < static_cast<int>(frames.size()); ++index) {
        DetectionResult detection = DetectSubtitleMask(frames[index], roi_rect);
        if (write_debug && detection.masked_pixels > 0 && index % 24 == 0) {
            cv::Mat overlay = frames[index].clone();
            for (const cv::Rect& box : detection.boxes) {
                cv::rectangle(overlay, box, cv::Scalar(0, 255, 0), 2);
            }

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(detection.mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            cv::drawContours(overlay, contours, -1, cv::Scalar(0, 0, 255), 1);

            std::ostringstream name;
            name << "mask_" << std::setw(4) << std::setfill('0') << index << ".jpg";
            cv::imwrite((fs::path(options.debug_dir) / name.str()).string(), overlay);
        }

        masks.push_back(detection.mask);
        detections.push_back(std::move(detection));
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
    if (write_debug) {
        std::cout << "Debug frames written to: " << options.debug_dir << '\n';
    }
    return 0;
}

}  // namespace remove_subtitle
