#include "remove_subtitle/options.hpp"
#include "remove_subtitle/processor.hpp"

#include <cstdlib>
#include <filesystem>

#include <gtest/gtest.h>
#include <opencv2/videoio.hpp>

namespace fs = std::filesystem;

TEST(CliE2ETest, ProcessesRepositoryTestVideo) {
    const char* run_slow = std::getenv("REMOVE_SUBTITLE_RUN_SLOW_TESTS");
    if (run_slow == nullptr || std::string(run_slow) != "1") {
        GTEST_SKIP() << "Set REMOVE_SUBTITLE_RUN_SLOW_TESTS=1 to run the full video pipeline test.";
    }

    const fs::path root = fs::path(PROJECT_SOURCE_DIR);
    const fs::path input = root / "tests/data/359.mp4";
    const fs::path output = root / "tests/data/359_test_output.mp4";

    remove_subtitle::Options options;
    options.input_path = input.string();
    options.output_path = output.string();

    if (fs::exists(output)) {
        fs::remove(output);
    }

    ASSERT_EQ(remove_subtitle::ProcessVideo(options), 0);
    ASSERT_TRUE(fs::exists(output));

    cv::VideoCapture capture(output.string());
    ASSERT_TRUE(capture.isOpened());
    EXPECT_GT(capture.get(cv::CAP_PROP_FRAME_COUNT), 0);
    EXPECT_EQ(static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH)), 720);
    EXPECT_EQ(static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT)), 1280);

    capture.release();
    fs::remove(output);
}
