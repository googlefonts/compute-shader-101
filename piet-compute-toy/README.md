# Compute shader toy based on piet-gpu

To run it without changing the shaders, just do `cargo run`.

To change the shaders, you will need to recompile those. You need [ninja] and spirv tools (glslangValidator and spirv-cross) in your path. The easiest way to get those is to install the [Vulkan SDK]. Then, to recompile the shaders, do:

```shell
(cd shader && ninja) && cargo run
```

This version is based on piet-gpu-hal, which is the runtime for [piet-gpu]. It's still very new so there are limitations and things that don't work yet. If you run into something, please file an issue!

[Vulkan SDK]: https://www.lunarg.com/vulkan-sdk/
[ninja]: https://ninja-build.org/
[piet-gpu]: https://github.com/linebender/piet-gpu
