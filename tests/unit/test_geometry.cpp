#include "remove_subtitle/geometry.hpp"

#include <gtest/gtest.h>

TEST(GeometryTest, ClampRectKeepsIntersectionWithinFrame) {
    const cv::Rect rect(-10, 20, 50, 40);
    const cv::Size frame_size(100, 80);

    const cv::Rect clamped = remove_subtitle::ClampRect(rect, frame_size);

    EXPECT_EQ(clamped.x, 0);
    EXPECT_EQ(clamped.y, 20);
    EXPECT_EQ(clamped.width, 40);
    EXPECT_EQ(clamped.height, 40);
}
