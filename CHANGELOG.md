# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.23.0] - 2023-04-29
### Updated
- Target wgpu 0.16

## [0.22.0] - 2023-02-16
### Updated
- Target egui 0.21

## [0.21.0] - 2023-02-01
### Updated
- Target egui 0.20
- Target wgpu 0.15

## [0.20.0] - 2022-10-07
### Updated
- Target wgpu 0.14

## [0.19.0] - 2022-08-26
### Updated
- Target egui 0.19

## [0.18.0] - 2022-07-03
### Updated
- Target egui 0.18
- Target wgpu 0.13

## [0.17.0] - 2022-03-09
### Updated
- Target egui 0.17
- `update_textures` and `update_user_textures` replaced by `add_textures` (to be called before
  `execute`) and `remove_textures` (to be called after `execute`).
- Enables the `convert_bytemuck` feature on egui to avoid internal use of unsafe.

## [0.16.0] - 2021-12-31
### Updated
- Target egui 0.16

## [0.15.0] - 2021-12-18
### Added
- `execute_with_renderpass`, allowing rendering egui onto an existing renderpass.
- `egui_texture_from_wgpu_texture_with_sampler_options`, allowing custom sampler options.

### Deprecated
- `web` feature is now a no-op, srgb-ness will be derived from output format.

### Updated
- Target wgpu 0.12

## [0.14.0] - 2021-10-27
### Updated
- Target egui 0.15

## [0.13.0] - 2021-10-09
### Updated
- Only target wgpu 0.11

## [0.12.1] - 2021-10-08
### Updated
- Target wgpu 0.10 and 0.11

## [0.12.0] - 2021-08-27
### Updated
- Target egui 0.14

## [0.11.0] - 2021-08-19
### Updated
- Target wgpu 0.10
- Allow replacing wgpu::Texture for a given egui::TextureId.
- Reduce panics.

## [0.10.0] - 2021-06-26
### Updated
- Target egui 0.13
- Update bytemuck dependency.

## [0.9.0] - 2021-06-20
### Updated
- Target wgpu 0.9
- Port shaders to WGSL.
- Add sample count to RenderPass.
- Allow setting the texture filter mode for user textures.

## [0.8.0] - 2021-05-11
### Updated
- Target egui 0.12

## [0.7.0] - 2021-05-01
### Updated
- Target wgpu 0.8

## [0.6.0] - 2021-04-13
### Updated
- Target egui 0.11
- Set "strip_index_format" to None.

## [0.5.0] - 2021-03-16
### Updated
- Target egui 0.10
- Fix sRGB color font handling (web + native)

### Added
- Added a function to use off-screen textures inside the egui UI.

## [0.4.0] - 2021-02-01
### Updated
- Target egui 0.8.
- Target wgpu 0.7.

## [0.3.0] - 2020-12-17
### Updated
- Target egui 0.5.

## [0.2.2] - 2020-11-16
### Updated
- Removed the mutability on GPU resources.
- Some clippy cleanups.
- Updated dependencies.

## [0.2.1] - 2020-11-13
### Updated
- Switch to linear texture filtering for the default sampler.

## [0.2.0] - 2020-11-13
### Added
- Add support for user textures.
### Updated
- Target egui 0.3.

## [0.1.0] - 2020-10-12
### Added
- Initial commit.
