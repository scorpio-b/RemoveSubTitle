#include "remove_subtitle/processor.hpp"

#include "remove_subtitle/geometry.hpp"
#include "remove_subtitle/region_restorer.hpp"
#include "remove_subtitle/subtitle_detector.hpp"

#include <filesystem>
#include <iomanip>
#include <iostream>
#include <atomic>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace remove_subtitle {

namespace {

std::mutex& LogMutex() {
    static std::mutex mutex;
    return mutex;
}

void LogLine(const std::string& message) {
    std::lock_guard<std::mutex> lock(LogMutex());
    std::cout << message << std::endl;
}

template <typename Fn>
void ParallelFor(std::size_t item_count, std::size_t thread_count, Fn&& fn) {
    if (item_count == 0) {
        return;
    }

    const std::size_t workers = std::max<std::size_t>(1, std::min(thread_count, item_count));
    if (workers == 1) {
        for (std::size_t index = 0; index < item_count; ++index) {
            fn(index);
        }
        return;
    }

    std::atomic<std::size_t> next_index{0};
    std::vector<std::thread> threads;
    threads.reserve(workers);

    for (std::size_t worker = 0; worker < workers; ++worker) {
        threads.emplace_back([&]() {
            while (true) {
                const std::size_t index = next_index.fetch_add(1);
                if (index >= item_count) {
                    break;
                }
                fn(index);
            }
        });
    }

    for (std::thread& thread : threads) {
        thread.join();
    }
}

std::vector<cv::Mat> StabilizeMasksTemporally(
    const std::vector<DetectionResult>& detections,
    const cv::Rect& roi_rect
) {
    std::vector<cv::Mat> stabilized_masks;
    stabilized_masks.reserve(detections.size());

    for (int index = 0; index < static_cast<int>(detections.size()); ++index) {
        cv::Mat stabilized = detections[index].mask.clone();
        if (cv::countNonZero(stabilized) == 0) {
            stabilized_masks.push_back(stabilized);
            continue;
        }

        cv::Mat support = cv::Mat::zeros(stabilized.size(), CV_8UC1);
        int supporting_neighbors = 0;

        for (int offset = 1; offset <= 2; ++offset) {
            const int prev = index - offset;
            if (prev >= 0) {
                cv::Mat overlap;
                cv::bitwise_and(detections[index].band_mask, detections[prev].mask, overlap);
                if (cv::countNonZero(overlap) > 0) {
                    cv::bitwise_or(support, overlap, support);
                    ++supporting_neighbors;
                }
            }

            const int next = index + offset;
            if (next < static_cast<int>(detections.size())) {
                cv::Mat overlap;
                cv::bitwise_and(detections[index].band_mask, detections[next].mask, overlap);
                if (cv::countNonZero(overlap) > 0) {
                    cv::bitwise_or(support, overlap, support);
                    ++supporting_neighbors;
                }
            }
        }

        if (supporting_neighbors == 0) {
            stabilized.setTo(0);
            stabilized_masks.push_back(stabilized);
            continue;
        }

        cv::Mat roi_support = support(roi_rect).clone();
        cv::Mat dilate_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 5));
        cv::dilate(roi_support, roi_support, dilate_kernel, cv::Point(-1, -1), 1);

        cv::Mat roi_current = stabilized(roi_rect).clone();
        cv::bitwise_and(roi_current, roi_support, roi_current);
        stabilized.setTo(0);
        roi_current.copyTo(stabilized(roi_rect));

        stabilized_masks.push_back(stabilized);
    }

    return stabilized_masks;
}

}  // namespace

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

    cv::setNumThreads(1);

    LogLine("Stage: load frames");
    std::vector<cv::Mat> frames;
    for (cv::Mat frame; capture.read(frame);) {
        frames.push_back(frame.clone());
    }
    if (frames.empty()) {
        throw std::runtime_error("no frames read from input video");
    }
    LogLine("Loaded frames: " + std::to_string(frames.size()));

    std::vector<cv::Mat> masks;
    std::vector<DetectionResult> detections;
    masks.resize(frames.size());
    detections.resize(frames.size());
    LogLine("Stage: detect subtitle masks");
    std::atomic<std::size_t> detected_count{0};
    ParallelFor(frames.size(), options.thread_count, [&](std::size_t index) {
        DetectionResult detection = DetectSubtitleMask(frames[index], roi_rect);
        masks[index] = detection.mask;
        detections[index] = std::move(detection);

        const std::size_t done = detected_count.fetch_add(1) + 1;
        if (done == frames.size() || done % 24 == 0) {
            LogLine("Detect progress: " + std::to_string(done) + "/" + std::to_string(frames.size()));
        }
    });

    LogLine("Stage: stabilize masks");
    masks = StabilizeMasksTemporally(detections, roi_rect);

    int subtitle_frames = 0;
    if (write_debug) {
        for (int index = 0; index < static_cast<int>(frames.size()); ++index) {
            if (cv::countNonZero(masks[index]) == 0 || index % 24 != 0) {
                continue;
            }

            cv::Mat overlay = frames[index].clone();
            for (const cv::Rect& box : detections[index].boxes) {
                cv::rectangle(overlay, box, cv::Scalar(0, 255, 0), 2);
            }

            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(masks[index], contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
            cv::drawContours(overlay, contours, -1, cv::Scalar(0, 0, 255), 1);

            std::ostringstream name;
            name << "mask_" << std::setw(4) << std::setfill('0') << index << ".jpg";
            cv::imwrite((fs::path(options.debug_dir) / name.str()).string(), overlay);
        }
    }

    const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    cv::VideoWriter writer(options.output_path, fourcc, fps, frame_size, true);
    if (!writer.isOpened()) {
        throw std::runtime_error("failed to open output video");
    }

    std::vector<cv::Mat> restored_frames(frames.size());
    LogLine("Stage: restore subtitle regions");
    std::atomic<std::size_t> restored_count{0};
    ParallelFor(frames.size(), options.thread_count, [&](std::size_t index) {
        const bool has_subtitle = cv::countNonZero(masks[index]) > 0;
        restored_frames[index] = has_subtitle
            ? RestoreFrame(
                frames,
                masks,
                static_cast<int>(index),
                options.temporal_window,
                options.patch_radius
            )
            : frames[index];

        const std::size_t done = restored_count.fetch_add(1) + 1;
        if (done == frames.size() || done % 12 == 0) {
            LogLine("Restore progress: " + std::to_string(done) + "/" + std::to_string(frames.size()));
        }
    });

    LogLine("Stage: write output video");
    for (std::size_t index = 0; index < frames.size(); ++index) {
        subtitle_frames += cv::countNonZero(masks[index]) > 0 ? 1 : 0;
        writer.write(restored_frames[index]);
        if (index + 1 == frames.size() || (index + 1) % 24 == 0) {
            LogLine("Write progress: " + std::to_string(index + 1) + "/" + std::to_string(frames.size()));
        }
    }

    std::cout << "Processed video written to: " << options.output_path << '\n';
    std::cout << "Subtitle ROI: "
              << roi_rect.x << "," << roi_rect.y << ","
              << roi_rect.width << "," << roi_rect.height << '\n';
    std::cout << "Frames with subtitle mask: " << subtitle_frames << "/" << frames.size() << '\n';
    std::cout << "Thread count: " << options.thread_count << '\n';
    if (write_debug) {
        std::cout << "Debug frames written to: " << options.debug_dir << '\n';
    }
    return 0;
}

}  // namespace remove_subtitle
