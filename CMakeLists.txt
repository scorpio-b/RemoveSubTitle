cmake_minimum_required(VERSION 3.16)
project(RemoveSubtitle LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED)

add_executable(remove_subtitle
    src/main.cpp
)

target_include_directories(remove_subtitle PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(remove_subtitle PRIVATE ${OpenCV_LIBS})
