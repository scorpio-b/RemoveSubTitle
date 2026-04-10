#include "remove_subtitle/region_restorer.hpp"

#include "remove_subtitle/flow_aligner.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <vector>

namespace remove_subtitle {

namespace {

struct AlignedReference {
    int frame_index;
    int offset;
    cv::Mat warped_frame;
};

struct ReferenceChoice {
    cv::Vec3d color_sum = cv::Vec3d(0.0, 0.0, 0.0);
    double weight_sum = 0.0;
    int sample_count = 0;
};

struct CandidateReference {
    const AlignedReference* reference = nullptr;
    double score = std::numeric_limits<double>::max();
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

bool HasReliableReferencePixel(
    const std::vector<cv::Mat>& masks,
    const AlignedReference& reference,
    int y,
    int x
) {
    return masks[reference.frame_index].at<uchar>(y, x) == 0;
}

std::vector<CandidateReference> SelectBestReferences(
    const std::vector<AlignedReference>& aligned_references,
    const std::vector<cv::Mat>& masks,
    const std::vector<cv::Mat>& frames,
    int index,
    const cv::Mat& current_mask,
    const cv::Rect& expanded
) {
    std::vector<CandidateReference> candidates;
    candidates.reserve(aligned_references.size());

    for (const AlignedReference& reference : aligned_references) {
        const double score =
            BorderDifference(frames[index], reference.warped_frame, current_mask, expanded) +
            reference.offset * 1.25;
        if (score >= 125.0) {
            continue;
        }

        int reliable_pixels = 0;
        int total_masked = 0;
        for (int y = expanded.y; y < expanded.y + expanded.height; ++y) {
            for (int x = expanded.x; x < expanded.x + expanded.width; ++x) {
                if (current_mask.at<uchar>(y, x) == 0) {
                    continue;
                }
                ++total_masked;
                reliable_pixels += HasReliableReferencePixel(masks, reference, y, x) ? 1 : 0;
            }
        }

        if (total_masked == 0) {
            continue;
        }
        const double reliable_ratio =
            static_cast<double>(reliable_pixels) / static_cast<double>(total_masked);
        if (reliable_ratio < 0.35) {
            continue;
        }

        candidates.push_back({&reference, score});
    }

    std::sort(
        candidates.begin(),
        candidates.end(),
        [](const CandidateReference& lhs, const CandidateReference& rhs) {
            return lhs.score < rhs.score;
        }
    );

    if (candidates.size() > 4) {
        candidates.resize(4);
    }
    return candidates;
}

cv::Mat InpaintBaseFrame(const cv::Mat& frame, const cv::Mat& mask) {
    cv::Mat inpainted;
    cv::inpaint(frame, mask, inpainted, 5.0, cv::INPAINT_TELEA);
    return inpainted;
}

void SmoothMaskedRegion(
    cv::Mat& frame,
    const cv::Mat& mask,
    const cv::Rect& region
) {
    const cv::Rect expanded = ExpandRect(region, 6, frame.size());
    cv::Mat roi = frame(expanded).clone();
    cv::Mat blurred;
    cv::bilateralFilter(roi, blurred, 7, 20.0, 20.0);
    blurred.copyTo(frame(expanded), mask(expanded));
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

cv::Vec3b TemporalSample(
    const std::vector<CandidateReference>& candidates,
    const std::vector<cv::Mat>& masks,
    int y,
    int x,
    bool& used_temporal
) {
    struct PixelSample {
        cv::Vec3b pixel;
        double weight;
        double luminance;
    };

    std::vector<PixelSample> samples;
    samples.reserve(candidates.size());
    ReferenceChoice choice;
    for (const CandidateReference& candidate : candidates) {
        const AlignedReference& reference = *candidate.reference;
        if (!HasReliableReferencePixel(masks, reference, y, x)) {
            continue;
        }

        const cv::Vec3b pixel = reference.warped_frame.at<cv::Vec3b>(y, x);
        const double weight = 1.0 / std::max(1.0, candidate.score);
        const double luminance =
            0.114 * pixel[0] + 0.587 * pixel[1] + 0.299 * pixel[2];
        samples.push_back({pixel, weight, luminance});
    }

    if (samples.size() < 2) {
        used_temporal = false;
        return {};
    }

    std::vector<double> luminances;
    luminances.reserve(samples.size());
    for (const PixelSample& sample : samples) {
        luminances.push_back(sample.luminance);
    }
    std::sort(luminances.begin(), luminances.end());
    const double median_luminance = luminances[luminances.size() / 2];

    for (const PixelSample& sample : samples) {
        if (sample.luminance + 12.0 < median_luminance) {
            continue;
        }
        choice.color_sum[0] += sample.pixel[0] * sample.weight;
        choice.color_sum[1] += sample.pixel[1] * sample.weight;
        choice.color_sum[2] += sample.pixel[2] * sample.weight;
        choice.weight_sum += sample.weight;
        ++choice.sample_count;
    }

    if (choice.sample_count == 0 || choice.weight_sum <= 0.0) {
        used_temporal = false;
        return {};
    }

    used_temporal = true;
    const double scale = 1.0 / choice.weight_sum;
    return cv::Vec3b(
        cv::saturate_cast<uchar>(choice.color_sum[0] * scale),
        cv::saturate_cast<uchar>(choice.color_sum[1] * scale),
        cv::saturate_cast<uchar>(choice.color_sum[2] * scale)
    );
}

cv::Mat RestoreFrame(
    const std::vector<cv::Mat>& frames,
    const std::vector<cv::Mat>& masks,
    int index,
    int temporal_window,
    int patch_radius
) {
    cv::Mat restored = InpaintBaseFrame(frames[index], masks[index]);
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
        const int region_padding = std::max(4, patch_radius + 4);
        const cv::Rect expanded = ExpandRect(region, region_padding, restored.size());
        const cv::Mat expanded_mask = current_mask(expanded);
        const std::vector<CandidateReference> candidates =
            SelectBestReferences(aligned_references, masks, frames, index, current_mask, expanded);

        for (int y = expanded.y; y < expanded.y + expanded.height; ++y) {
            for (int x = expanded.x; x < expanded.x + expanded.width; ++x) {
                if (current_mask.at<uchar>(y, x) == 0) {
                    continue;
                }

                bool used_temporal = false;
                const cv::Vec3b temporal_pixel =
                    TemporalSample(candidates, masks, y, x, used_temporal);
                restored.at<cv::Vec3b>(y, x) = used_temporal
                    ? temporal_pixel
                    : SpatialFallbackSample(restored, current_mask, y, x);
            }
        }

        SmoothMaskedRegion(restored, current_mask, expanded);
    }

    return restored;
}

}  // namespace remove_subtitle
