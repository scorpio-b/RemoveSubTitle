#include "remove_subtitle/options.hpp"

#include <iostream>
#include <stdexcept>

namespace remove_subtitle {

void PrintUsage() {
    std::cout
        << "Usage: remove_subtitle <input_video> <output_video> "
        << "[x y width height temporal_window patch_radius [debug_dir]]\n";
}

Options ParseArgs(int argc, char** argv) {
    if (argc != 3 && argc != 9 && argc != 10) {
        PrintUsage();
        throw std::runtime_error("invalid arguments");
    }

    Options options;
    options.input_path = argv[1];
    options.output_path = argv[2];

    if (argc >= 9) {
        options.x = std::stoi(argv[3]);
        options.y = std::stoi(argv[4]);
        options.width = std::stoi(argv[5]);
        options.height = std::stoi(argv[6]);
        options.temporal_window = std::stoi(argv[7]);
        options.patch_radius = std::stoi(argv[8]);
    }

    if (argc == 10) {
        options.debug_dir = argv[9];
    }

    return options;
}

}  // namespace remove_subtitle
