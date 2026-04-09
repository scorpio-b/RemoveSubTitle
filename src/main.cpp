#include "remove_subtitle/options.hpp"
#include "remove_subtitle/processor.hpp"

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
    try {
        const auto options = remove_subtitle::ParseArgs(argc, argv);
        return remove_subtitle::ProcessVideo(options);
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return 1;
    }
}
