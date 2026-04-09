# RemoveSubTitle

C++17 experiment for removing hardcoded subtitles from video using temporal restoration.
去掉即梦生成的 mp4 格式视频里的硬字幕。

## Build

```bash
cmake -S . -B build
cmake --build build -j4
```

## Run

```bash
./build/remove_subtitle tests/data/359.mp4 output.mp4
```

Optional parameters:

```text
remove_subtitle <input_video> <output_video> [x y width height temporal_window patch_radius]
```

## Tests

Tests are disabled by default so IDE configuration does not depend on downloading GoogleTest.

```bash
cmake -S . -B build -DREMOVE_SUBTITLE_BUILD_TESTS=ON
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

## Test Asset

The repository keeps one local test video at `tests/data/359.mp4`.
