# egui_wgpu_backend

[![Latest version](https://img.shields.io/crates/v/egui_wgpu_backend.svg)](https://crates.io/crates/egui_wgpu_backend)
[![Documentation](https://docs.rs/egui_wgpu_backend/badge.svg)](https://docs.rs/egui_wgpu_backend)
![MIT](https://img.shields.io/badge/license-MIT-blue.svg)
![Apache](https://img.shields.io/badge/license-Apache-blue.svg)

Backend code to run [egui](https://github.com/emilk/egui) using [wgpu](https://wgpu.rs/).

## Features

 * `web` Using this features will force the backend to output sRGB gamma encoded colors. Normally
   shaders are supposed to work in linear space, but browsers want sRGBA gamma encoded colors instead.

## Example
We have created [a simple example](https://github.com/hasenbanck/egui_example) project to show you, how to use this crate.

## License
egui_wgpu_backend is distributed under the terms of both the MIT license and the Apache License (Version 2.0).

See [LICENSE-APACHE](LICENSE-APACHE), [LICENSE-MIT](LICENSE-MIT).
