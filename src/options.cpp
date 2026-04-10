#include "remove_subtitle/options.hpp"

#include <iostream>
#include <thread>
#include <stdexcept>

namespace remove_subtitle {

void PrintUsage() {
    std::cout
        << "Usage: remove_subtitle <input_video> <output_video> "
        << "[x y width height temporal_window patch_radius [debug_dir [thread_count [repair_expand]]]]\n";
}

Options ParseArgs(int argc, char** argv) {
    if (argc != 3 && argc != 9 && argc != 10 && argc != 11 && argc != 12) {
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

    if (argc >= 10) {
        options.debug_dir = argv[9];
    }

    if (argc >= 11) {
        options.thread_count = static_cast<std::size_t>(std::stoul(argv[10]));
    }

    if (argc == 12) {
        options.repair_expand = std::stoi(argv[11]);
    }

    if (options.repair_expand < 0) {
        throw std::runtime_error("repair_expand must be non-negative");
    }

    if (options.thread_count == 0) {
        const unsigned int hardware_threads = std::thread::hardware_concurrency();
        options.thread_count = hardware_threads == 0 ? 4 : hardware_threads;
    }

    return options;
}

}  // namespace remove_subtitle
