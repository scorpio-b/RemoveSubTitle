#pragma once

#include <cstddef>
#include <string>

namespace remove_subtitle {

struct Options {
    std::string input_path;
    std::string output_path;
    std::string debug_dir;
    int x = 180;
    int y = 820;
    int width = 360;
    int height = 220;
    int temporal_window = 12;
    int patch_radius = 2;
    std::size_t thread_count = 0;
};

Options ParseArgs(int argc, char** argv);
void PrintUsage();

}  // namespace remove_subtitle
