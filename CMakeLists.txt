cmake_minimum_required(VERSION 3.16)
project(RemoveSubtitle LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(REMOVE_SUBTITLE_BUILD_TESTS "Build GoogleTest-based test targets" OFF)

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(TESSERACT REQUIRED tesseract lept)

set(REMOVE_SUBTITLE_SOURCES
    src/flow_aligner.cpp
    src/geometry.cpp
    src/options.cpp
    src/processor.cpp
    src/region_restorer.cpp
    src/subtitle_detector.cpp
)

add_library(remove_subtitle_lib ${REMOVE_SUBTITLE_SOURCES})
target_include_directories(remove_subtitle_lib PUBLIC include ${OpenCV_INCLUDE_DIRS} ${TESSERACT_INCLUDE_DIRS})
target_link_directories(remove_subtitle_lib PUBLIC ${TESSERACT_LIBRARY_DIRS})
target_link_libraries(remove_subtitle_lib PUBLIC ${OpenCV_LIBS} ${TESSERACT_LIBRARIES})

add_executable(remove_subtitle src/main.cpp)
target_link_libraries(remove_subtitle PRIVATE remove_subtitle_lib)

if(REMOVE_SUBTITLE_BUILD_TESTS)
    include(CTest)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()

    add_executable(remove_subtitle_tests
        tests/e2e/test_cli.cpp
        tests/unit/test_detector.cpp
        tests/unit/test_flow.cpp
        tests/unit/test_geometry.cpp
        tests/unit/test_restorer.cpp
    )
    target_link_libraries(remove_subtitle_tests
        PRIVATE
        remove_subtitle_lib
        GTest::gtest_main
    )
    target_include_directories(remove_subtitle_tests PRIVATE include ${OpenCV_INCLUDE_DIRS})
    target_compile_definitions(remove_subtitle_tests PRIVATE PROJECT_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

    include(GoogleTest)
    gtest_discover_tests(remove_subtitle_tests)
endif()
