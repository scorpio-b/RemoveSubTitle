# RemoveSubTitle

使用时间域修复技术去除视频硬字幕的 C++17 实验项目。
主要用于去掉即梦生成的 mp4 格式视频里的硬字幕。

## 编译

```bash
cmake -S . -B build
cmake --build build -j4
```

## 运行

```bash
./build/remove_subtitle tests/data/359.mp4 output.mp4
```

可选参数：

```text
remove_subtitle <input_video> <output_video> [x y width height temporal_window patch_radius]
```

完整形式：

```text
remove_subtitle <input_video> <output_video> [x y width height temporal_window patch_radius [debug_dir [thread_count]]]
```

示例：

```bash
./build/remove_subtitle \
  /Users/yutao/MyWorks/RemoveSubTitle/tests/data/359.mp4 \
  /Users/yutao/MyWorks/RemoveSubTitle/artifacts/output/359_out.mp4 \
  180 820 360 220 12 2 \
  /Users/yutao/MyWorks/RemoveSubTitle/artifacts/debug \
  12
```

参数说明：

- `input_video`: 输入的 mp4 文件
- `output_video`: 输出视频路径
- `x y width height`: 字幕区域坐标
- `temporal_window`: 两侧相邻帧的搜索范围
- `patch_radius`: 修复时 patch 比对的半径
- `debug_dir`: 可选目录，用于输出调试掩码叠加图
- `thread_count`: 可选工作线程数，默认为硬件并发数

建议的输出目录：

- 输出视频：`artifacts/output/`
- 调试叠加图：`artifacts/debug/`

进度输出说明：

- `Stage: load frames`: 正在将所有帧加载到内存
- `Stage: detect subtitle masks`: 正在检测每帧的字幕掩码
- `Stage: stabilize masks`: 正在对掩码进行时域滤波
- `Stage: restore subtitle regions`: 正在修复字幕区域
- `Stage: write output video`: 正在编码输出视频

当程序打印类似以下内容时：

```text
Restore progress: 132/361
```

表示已完成 132 / 361 帧的字幕区域修复。

## CLion

推荐的运行配置：

- `Program`

```text
/Users/yutao/MyWorks/RemoveSubTitle/cmake-build-debug/remove_subtitle
```

- `Arguments`

```text
/Users/yutao/MyWorks/RemoveSubTitle/tests/data/359.mp4 /Users/yutao/MyWorks/RemoveSubTitle/artifacts/output/359_out.mp4 180 820 360 220 12 2 /Users/yutao/MyWorks/RemoveSubTitle/artifacts/debug 12
```

为了更快迭代，可以先裁剪一个短片段：

```bash
ffmpeg -y -ss 6 -t 4 -i /Users/yutao/MyWorks/RemoveSubTitle/tests/data/359.mp4 -c copy /tmp/359_clip.mp4
```

然后运行：

```text
/tmp/359_clip.mp4 /Users/yutao/MyWorks/RemoveSubTitle/artifacts/output/359_clip_out.mp4 180 820 360 220 12 2 /Users/yutao/MyWorks/RemoveSubTitle/artifacts/debug 12
```

## 测试

测试默认禁用，以避免 IDE 配置依赖下载 GoogleTest。

```bash
cmake -S . -B build -DREMOVE_SUBTITLE_BUILD_TESTS=ON
cmake --build build -j4
ctest --test-dir build --output-on-failure
```

## 测试素材

仓库在 `tests/data/359.mp4` 保留了一份本地测试视频。
