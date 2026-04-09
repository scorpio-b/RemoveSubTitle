#include "remove_subtitle/region_restorer.hpp"

#include "remove_subtitle/flow_aligner.hpp"

#include <algorithm>
#include <vector>

namespace remove_subtitle {

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
            aligned_references.push_back({prev, offset, WarpReferenceToCurrent(frames[index], frames[prev])});
        }

        const int next = index + offset;
        if (next < static_cast<int>(frames.size())) {
            aligned_references.push_back({next, offset, WarpReferenceToCurrent(frames[index], frames[next])});
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

            const double score =
                PatchDifference(frames[index], reference.warped_frame, pt, patch_radius) +
                reference.offset * 0.35;

            if (score > 28.0) {
                continue;
            }

            candidates.push_back({score, reference.warped_frame.at<cv::Vec3b>(pt)});
        }

        if (!candidates.empty()) {
            std::sort(
                candidates.begin(),
                candidates.end(),
                [](const Candidate& lhs, const Candidate& rhs) { return lhs.score < rhs.score; }
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

}  // namespace remove_subtitle
