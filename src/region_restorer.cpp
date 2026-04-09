#include "remove_subtitle/region_restorer.hpp"

#include "remove_subtitle/flow_aligner.hpp"

#include <algorithm>
#include <limits>
#include <vector>

namespace remove_subtitle {

namespace {

struct AlignedReference {
    int frame_index;
    int offset;
    cv::Mat warped_frame;
};

cv::Rect ExpandRect(const cv::Rect& rect, int padding, const cv::Size& bounds) {
    const int x = std::max(0, rect.x - padding);
    const int y = std::max(0, rect.y - padding);
    const int right = std::min(bounds.width, rect.x + rect.width + padding);
    const int bottom = std::min(bounds.height, rect.y + rect.height + padding);
    return cv::Rect(x, y, right - x, bottom - y);
}

double BorderDifference(
    const cv::Mat& current,
    const cv::Mat& reference,
    const cv::Mat& mask,
    const cv::Rect& rect
) {
    const cv::Rect expanded = ExpandRect(rect, 2, current.size());
    double total = 0.0;
    int count = 0;

    for (int y = expanded.y; y < expanded.y + expanded.height; ++y) {
        for (int x = expanded.x; x < expanded.x + expanded.width; ++x) {
            const bool inside_rect =
                x >= rect.x && x < rect.x + rect.width &&
                y >= rect.y && y < rect.y + rect.height;
            if (inside_rect) {
                continue;
            }
            if (mask.at<uchar>(y, x) != 0) {
                continue;
            }

            const cv::Vec3b current_pixel = current.at<cv::Vec3b>(y, x);
            const cv::Vec3b reference_pixel = reference.at<cv::Vec3b>(y, x);
            total += std::abs(current_pixel[0] - reference_pixel[0]) +
                     std::abs(current_pixel[1] - reference_pixel[1]) +
                     std::abs(current_pixel[2] - reference_pixel[2]);
            ++count;
        }
    }

    if (count == 0) {
        return std::numeric_limits<double>::max();
    }

    return total / static_cast<double>(count);
}

std::vector<cv::Rect> ExtractMaskedRegions(const cv::Mat& mask) {
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int component_count = cv::connectedComponentsWithStats(mask, labels, stats, centroids, 8);

    std::vector<cv::Rect> regions;
    for (int label = 1; label < component_count; ++label) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area < 6) {
            continue;
        }

        const int left = stats.at<int>(label, cv::CC_STAT_LEFT);
        const int top = stats.at<int>(label, cv::CC_STAT_TOP);
        const int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
        const int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
        regions.emplace_back(left, top, width, height);
    }

    return regions;
}

}  // namespace

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

    if (cv::countNonZero(current_mask) == 0) {
        return restored;
    }

    std::vector<AlignedReference> aligned_references;
    aligned_references.reserve(temporal_window * 2);
    for (int offset = 1; offset <= temporal_window; ++offset) {
        const int prev = index - offset;
        if (prev >= 0) {
            aligned_references.push_back({prev, offset, WarpReferenceToCurrent(frames[index], frames[prev])});
        }

        const int next = index + offset;
        if (next < static_cast<int>(frames.size())) {
            aligned_references.push_back({next, offset, WarpReferenceToCurrent(frames[index], frames[next])});
        }
    }

    const std::vector<cv::Rect> regions = ExtractMaskedRegions(current_mask);
    for (const cv::Rect& region : regions) {
        const cv::Rect expanded = ExpandRect(region, patch_radius + 2, restored.size());

        double best_score = std::numeric_limits<double>::max();
        const cv::Mat* best_reference = nullptr;
        for (const AlignedReference& reference : aligned_references) {
            const cv::Point center(region.x + region.width / 2, region.y + region.height / 2);
            if (masks[reference.frame_index].at<uchar>(center) != 0) {
                continue;
            }

            const double score =
                BorderDifference(frames[index], reference.warped_frame, current_mask, expanded) +
                reference.offset * 1.25;
            if (score < best_score) {
                best_score = score;
                best_reference = &reference.warped_frame;
            }
        }

        if (best_reference != nullptr && best_score < 90.0) {
            cv::Mat target_roi = restored(expanded);
            cv::Mat reference_roi = (*best_reference)(expanded);
            cv::Mat mask_roi = current_mask(expanded);

            reference_roi.copyTo(target_roi, mask_roi);
            continue;
        }

        for (int y = region.y; y < region.y + region.height; ++y) {
            for (int x = region.x; x < region.x + region.width; ++x) {
                if (current_mask.at<uchar>(y, x) == 0) {
                    continue;
                }
                restored.at<cv::Vec3b>(y, x) = SpatialFallbackSample(restored, current_mask, y, x);
            }
        }
    }

    cv::Mat blur_source = restored.clone();
    cv::GaussianBlur(blur_source, blur_source, cv::Size(3, 3), 0.0);
    blur_source.copyTo(restored, current_mask);
    return restored;
}

}  // namespace remove_subtitle
