#include "remove_subtitle/flow_aligner.hpp"

#include <gtest/gtest.h>

TEST(FlowAlignerTest, PatchDifferenceIsZeroForIdenticalPatch) {
    cv::Mat frame(10, 10, CV_8UC3, cv::Scalar(20, 40, 60));
    const double diff = remove_subtitle::PatchDifference(frame, frame, cv::Point(5, 5), 2);
    EXPECT_DOUBLE_EQ(diff, 0.0);
}
