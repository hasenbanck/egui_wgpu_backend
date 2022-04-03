//! A render backend to use [egui](https://github.com/emilk/egui) with [wgpu](https://github.com/gfx-rs/wgpu-rs).
//!
//! You need to create a [`RenderPass`] and feed it with the output data provided by egui.
//! A basic usage example can be found [here](https://github.com/hasenbanck/egui_example).
#![warn(missing_docs)]

use std::{
    borrow::Cow,
    collections::{hash_map::Entry, HashMap},
    fmt::Formatter,
    num::NonZeroU32,
};

use bytemuck::{Pod, Zeroable};
pub use wgpu;
use wgpu::util::DeviceExt;

/// Error that the backend can return.
#[derive(Debug)]
pub enum BackendError {
    /// The given `egui::TextureId` was invalid.
    InvalidTextureId(String),
    /// Internal implementation error.
    Internal(String),
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendError::InvalidTextureId(msg) => {
                write!(f, "invalid TextureId: `{:?}`", msg)
            }
            BackendError::Internal(msg) => {
                write!(f, "internal error: `{:?}`", msg)
            }
        }
    }
}

impl std::error::Error for BackendError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}

/// Enum for selecting the right buffer type.
#[derive(Debug)]
enum BufferType {
    Uniform,
    Index,
    Vertex,
}

/// Information about the screen used for rendering.
pub struct ScreenDescriptor {
    /// Width of the window in physical pixel.
    pub physical_width: u32,
    /// Height of the window in physical pixel.
    pub physical_height: u32,
    /// HiDPI scale factor.
    pub scale_factor: f32,
}

impl ScreenDescriptor {
    fn logical_size(&self) -> (u32, u32) {
        let logical_width = self.physical_width as f32 / self.scale_factor;
        let logical_height = self.physical_height as f32 / self.scale_factor;
        (logical_width as u32, logical_height as u32)
    }
}

/// Uniform buffer used when rendering.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct UniformBuffer {
    screen_size: [f32; 2],
    // Without this padding, rendering would fail for the WebGL backend due to the minimum uniform buffer size of 16 bytes.
    // See https://github.com/hasenbanck/egui_wgpu_backend/issues/58
    _padding: [f32; 2],
}

unsafe impl Pod for UniformBuffer {}

unsafe impl Zeroable for UniformBuffer {}

/// Wraps the buffers and includes additional information.
#[derive(Debug)]
struct SizedBuffer {
    buffer: wgpu::Buffer,
    size: usize,
}

/// RenderPass to render a egui based GUI.
pub struct RenderPass {
    render_pipeline: wgpu::RenderPipeline,
    index_buffers: Vec<SizedBuffer>,
    vertex_buffers: Vec<SizedBuffer>,
    uniform_buffer: SizedBuffer,
    uniform_bind_group: wgpu::BindGroup,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    next_user_texture_id: u64,

    /// Map of egui texture IDs to textures and their associated bindgroups (texture view +
    /// sampler). The texture may be None if the TextureId is just a handle to a user-provided
    /// sampler.
    textures: HashMap<egui::TextureId, (Option<wgpu::Texture>, wgpu::BindGroup)>,
}

impl RenderPass {
    /// Creates a new render pass to render a egui UI.
    ///
    /// If the format passed is not a *Srgb format, the shader will automatically convert to sRGB colors in the shader.
    pub fn new(
        device: &wgpu::Device,
        output_format: wgpu::TextureFormat,
        msaa_samples: u32,
    ) -> Self {
        let shader = wgpu::ShaderModuleDescriptor {
            label: Some("egui_shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader/egui.wgsl"))),
        };
        let module = device.create_shader_module(&shader);

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("egui_uniform_buffer"),
            contents: bytemuck::cast_slice(&[UniformBuffer {
                screen_size: [0.0, 0.0],
                _padding: [0.0; 2],
            }]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let uniform_buffer = SizedBuffer {
            buffer: uniform_buffer,
            size: std::mem::size_of::<UniformBuffer>(),
        };

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("egui_uniform_bind_group_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        has_dynamic_offset: false,
                        min_binding_size: None,
                        ty: wgpu::BufferBindingType::Uniform,
                    },
                    count: None,
                }],
            });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("egui_uniform_bind_group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_buffer.buffer,
                    offset: 0,
                    size: None,
                }),
            }],
        });

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("egui_texture_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("egui_pipeline_layout"),
            bind_group_layouts: &[&uniform_bind_group_layout, &texture_bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("egui_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                entry_point: if output_format.describe().srgb {
                    "vs_main"
                } else {
                    "vs_conv_main"
                },
                module: &module,
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 5 * 4,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    // 0: vec2 position
                    // 1: vec2 texture coordinates
                    // 2: uint color
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2, 2 => Uint32],
                }],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                unclipped_depth: false,
                conservative: false,
                cull_mode: None,
                front_face: wgpu::FrontFace::default(),
                polygon_mode: wgpu::PolygonMode::default(),
                strip_index_format: None,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                alpha_to_coverage_enabled: false,
                count: msaa_samples,
                mask: !0,
            },

            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: "fs_main",
                targets: &[wgpu::ColorTargetState {
                    format: output_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::OneMinusDstAlpha,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                }],
            }),
            multiview: None,
        });

        Self {
            render_pipeline,
            vertex_buffers: Vec::with_capacity(64),
            index_buffers: Vec::with_capacity(64),
            uniform_buffer,
            uniform_bind_group,
            texture_bind_group_layout,
            next_user_texture_id: 0,
            textures: HashMap::new(),
        }
    }

    /// Executes the egui render pass. When `clear_on_draw` is set, the output target will get cleared before writing to it.
    pub fn execute(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        color_attachment: &wgpu::TextureView,
        paint_jobs: &[egui::epaint::ClippedMesh],
        screen_descriptor: &ScreenDescriptor,
        clear_color: Option<wgpu::Color>,
    ) -> Result<(), BackendError> {
        let load_operation = if let Some(color) = clear_color {
            wgpu::LoadOp::Clear(color)
        } else {
            wgpu::LoadOp::Load
        };

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[wgpu::RenderPassColorAttachment {
                view: color_attachment,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: load_operation,
                    store: true,
                },
            }],
            depth_stencil_attachment: None,
            label: Some("egui main render pass"),
        });
        rpass.push_debug_group("egui_pass");

        self.execute_with_renderpass(&mut rpass, paint_jobs, screen_descriptor)?;

        rpass.pop_debug_group();

        Ok(())
    }

    /// Executes the egui render pass onto an existing wgpu renderpass.
    pub fn execute_with_renderpass<'rpass>(
        &'rpass self,
        rpass: &mut wgpu::RenderPass<'rpass>,
        paint_jobs: &[egui::epaint::ClippedMesh],
        screen_descriptor: &ScreenDescriptor,
    ) -> Result<(), BackendError> {
        rpass.set_pipeline(&self.render_pipeline);

        rpass.set_bind_group(0, &self.uniform_bind_group, &[]);

        let scale_factor = screen_descriptor.scale_factor;
        let physical_width = screen_descriptor.physical_width;
        let physical_height = screen_descriptor.physical_height;

        for ((egui::ClippedMesh(clip_rect, mesh), vertex_buffer), index_buffer) in paint_jobs
            .iter()
            .zip(self.vertex_buffers.iter())
            .zip(self.index_buffers.iter())
        {
            // Transform clip rect to physical pixels.
            let clip_min_x = scale_factor * clip_rect.min.x;
            let clip_min_y = scale_factor * clip_rect.min.y;
            let clip_max_x = scale_factor * clip_rect.max.x;
            let clip_max_y = scale_factor * clip_rect.max.y;

            // Make sure clip rect can fit within an `u32`.
            let clip_min_x = clip_min_x.clamp(0.0, physical_width as f32);
            let clip_min_y = clip_min_y.clamp(0.0, physical_height as f32);
            let clip_max_x = clip_max_x.clamp(clip_min_x, physical_width as f32);
            let clip_max_y = clip_max_y.clamp(clip_min_y, physical_height as f32);

            let clip_min_x = clip_min_x.round() as u32;
            let clip_min_y = clip_min_y.round() as u32;
            let clip_max_x = clip_max_x.round() as u32;
            let clip_max_y = clip_max_y.round() as u32;

            let width = (clip_max_x - clip_min_x).max(1);
            let height = (clip_max_y - clip_min_y).max(1);

            {
                // Clip scissor rectangle to target size.
                let x = clip_min_x.min(physical_width);
                let y = clip_min_y.min(physical_height);
                let width = width.min(physical_width - x);
                let height = height.min(physical_height - y);

                // Skip rendering with zero-sized clip areas.
                if width == 0 || height == 0 {
                    continue;
                }

                rpass.set_scissor_rect(x, y, width, height);
            }
            let bind_group = self.get_texture_bind_group(mesh.texture_id)?;
            rpass.set_bind_group(1, bind_group, &[]);

            rpass.set_index_buffer(index_buffer.buffer.slice(..), wgpu::IndexFormat::Uint32);
            rpass.set_vertex_buffer(0, vertex_buffer.buffer.slice(..));
            rpass.draw_indexed(0..mesh.indices.len() as u32, 0, 0..1);
        }

        Ok(())
    }

    fn get_texture_bind_group(
        &self,
        texture_id: egui::TextureId,
    ) -> Result<&wgpu::BindGroup, BackendError> {
        self.textures
            .get(&texture_id)
            .ok_or_else(|| {
                BackendError::Internal(format!("Texture {:?} used but not live", texture_id))
            })
            .map(|x| &x.1)
    }

    /// Updates the texture used by egui for the fonts etc. Should be called before `execute()`.
    pub fn add_textures(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        textures: &egui::TexturesDelta,
    ) -> Result<(), BackendError> {
        for (texture_id, image_delta) in textures.set.iter() {
            let image_size = image_delta.image.size();

            let origin = match image_delta.pos {
                Some([x, y]) => wgpu::Origin3d {
                    x: x as u32,
                    y: y as u32,
                    z: 0,
                },
                None => wgpu::Origin3d::ZERO,
            };

            let alpha_srgb_pixels: Option<Vec<_>> = match &image_delta.image {
                egui::ImageData::Color(_) => None,
                egui::ImageData::Alpha(a) => Some(a.srgba_pixels(1.0).collect()),
            };

            let image_data: &[u8] = match &image_delta.image {
                egui::ImageData::Color(c) => bytemuck::cast_slice(c.pixels.as_slice()),
                egui::ImageData::Alpha(_) => {
                    // The unwrap here should never fail as alpha_srgb_pixels will have been set to
                    // `Some` above.
                    bytemuck::cast_slice(
                        alpha_srgb_pixels
                            .as_ref()
                            .expect("Alpha texture should have been converted already")
                            .as_slice(),
                    )
                }
            };

            let image_size = wgpu::Extent3d {
                width: image_size[0] as u32,
                height: image_size[1] as u32,
                depth_or_array_layers: 1,
            };

            let image_data_layout = wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: NonZeroU32::new(4 * image_size.width),
                rows_per_image: None,
            };

            let label_base = match texture_id {
                egui::TextureId::Managed(m) => format!("egui_image_{}", m),
                egui::TextureId::User(u) => format!("egui_user_image_{}", u),
            };

            match self.textures.entry(texture_id.clone()) {
                Entry::Occupied(mut o) => match image_delta.pos {
                    None => {
                        let (texture, bind_group) = create_texture_and_bind_group(
                            device,
                            queue,
                            &label_base,
                            origin,
                            image_data,
                            image_data_layout,
                            image_size,
                            &self.texture_bind_group_layout,
                        );

                        let (texture, _) = o.insert((Some(texture), bind_group));

                        if let Some(texture) = texture {
                            texture.destroy();
                        }
                    }
                    Some(_) => {
                        if let Some(texture) = o.get().0.as_ref() {
                            queue.write_texture(
                                wgpu::ImageCopyTexture {
                                    texture,
                                    mip_level: 0,
                                    origin,
                                    aspect: wgpu::TextureAspect::All,
                                },
                                image_data,
                                image_data_layout,
                                image_size,
                            );
                        } else {
                            return Err(BackendError::InvalidTextureId(format!(
                                "Update of unmanaged texture {:?}",
                                texture_id
                            )));
                        }
                    }
                },
                Entry::Vacant(v) => {
                    let (texture, bind_group) = create_texture_and_bind_group(
                        device,
                        queue,
                        &label_base,
                        origin,
                        image_data,
                        image_data_layout,
                        image_size,
                        &self.texture_bind_group_layout,
                    );

                    v.insert((Some(texture), bind_group));
                }
            }
        }

        Ok(())
    }

    /// Remove the textures egui no longer needs. Should be called after `execute()`
    pub fn remove_textures(&mut self, textures: egui::TexturesDelta) -> Result<(), BackendError> {
        for texture_id in textures.free {
            let (texture, _binding) = self.textures.remove(&texture_id).ok_or_else(|| {
                // This can happen due to a bug in egui, or if the user doesn't call `add_textures`
                // when required.
                BackendError::InvalidTextureId(format!(
                    "Attempted to remove an unknown texture {:?}",
                    texture_id
                ))
            })?;

            if let Some(texture) = texture {
                texture.destroy();
            }
        }

        Ok(())
    }

    /// Registers a `wgpu::Texture` with a `egui::TextureId`.
    ///
    /// This enables the application to reference the texture inside an image ui element.
    /// This effectively enables off-screen rendering inside the egui UI. Texture must have
    /// the texture format `TextureFormat::Rgba8UnormSrgb` and
    /// Texture usage `TextureUsage::SAMPLED`.
    pub fn egui_texture_from_wgpu_texture(
        &mut self,
        device: &wgpu::Device,
        texture: &wgpu::TextureView,
        texture_filter: wgpu::FilterMode,
    ) -> egui::TextureId {
        self.egui_texture_from_wgpu_texture_with_sampler_options(
            device,
            texture,
            wgpu::SamplerDescriptor {
                label: Some(
                    format!(
                        "egui_user_image_{}_texture_sampler",
                        self.next_user_texture_id
                    )
                    .as_str(),
                ),
                mag_filter: texture_filter,
                min_filter: texture_filter,
                ..Default::default()
            },
        )
    }

    /// Registers a `wgpu::Texture` with an existing `egui::TextureId`.
    ///
    /// This enables applications to reuse `TextureId`s.
    pub fn update_egui_texture_from_wgpu_texture(
        &mut self,
        device: &wgpu::Device,
        texture: &wgpu::TextureView,
        texture_filter: wgpu::FilterMode,
        id: egui::TextureId,
    ) -> Result<(), BackendError> {
        self.update_egui_texture_from_wgpu_texture_with_sampler_options(
            device,
            texture,
            wgpu::SamplerDescriptor {
                label: Some(
                    format!(
                        "egui_user_image_{}_texture_sampler",
                        self.next_user_texture_id
                    )
                    .as_str(),
                ),
                mag_filter: texture_filter,
                min_filter: texture_filter,
                ..Default::default()
            },
            id,
        )
    }

    /// Registers a `wgpu::Texture` with a `egui::TextureId` while also accepting custom
    /// `wgpu::SamplerDescriptor` options.
    ///
    /// This allows applications to specify individual minification/magnification filters as well as
    /// custom mipmap and tiling options.
    ///
    /// The `Texture` must have the format `TextureFormat::Rgba8UnormSrgb` and usage
    /// `TextureUsage::SAMPLED`. Any compare function supplied in the `SamplerDescriptor` will be
    /// ignored.
    pub fn egui_texture_from_wgpu_texture_with_sampler_options(
        &mut self,
        device: &wgpu::Device,
        texture: &wgpu::TextureView,
        sampler_descriptor: wgpu::SamplerDescriptor,
    ) -> egui::TextureId {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            compare: None,
            ..sampler_descriptor
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(
                format!(
                    "egui_user_image_{}_texture_bind_group",
                    self.next_user_texture_id
                )
                .as_str(),
            ),
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(texture),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let id = egui::TextureId::User(self.next_user_texture_id);
        self.textures.insert(id, (None, bind_group));
        self.next_user_texture_id += 1;

        id
    }

    /// Registers a `wgpu::Texture` with an existing `egui::TextureId` while also accepting custom
    /// `wgpu::SamplerDescriptor` options.
    ///
    /// This allows applications to reuse `TextureId`s created with custom sampler options.
    pub fn update_egui_texture_from_wgpu_texture_with_sampler_options(
        &mut self,
        device: &wgpu::Device,
        texture: &wgpu::TextureView,
        sampler_descriptor: wgpu::SamplerDescriptor,
        id: egui::TextureId,
    ) -> Result<(), BackendError> {
        if let egui::TextureId::Managed(_) = id {
            return Err(BackendError::InvalidTextureId(
                "ID was not of type `TextureId::User`".to_string(),
            ));
        }

        let (_user_texture, user_texture_binding) =
            self.textures.get_mut(&id).ok_or_else(|| {
                BackendError::InvalidTextureId(format!(
                    "user texture for TextureId {:?} could not be found",
                    id
                ))
            })?;

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            compare: None,
            ..sampler_descriptor
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(
                format!("egui_user_{}_texture_bind_group", self.next_user_texture_id).as_str(),
            ),
            layout: &self.texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(texture),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        *user_texture_binding = bind_group;

        Ok(())
    }

    /// Uploads the uniform, vertex and index data used by the render pass.
    /// Should be called before `execute()`.
    pub fn update_buffers(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        paint_jobs: &[egui::epaint::ClippedMesh],
        screen_descriptor: &ScreenDescriptor,
    ) {
        let index_size = self.index_buffers.len();
        let vertex_size = self.vertex_buffers.len();

        let (logical_width, logical_height) = screen_descriptor.logical_size();

        self.update_buffer(
            device,
            queue,
            BufferType::Uniform,
            0,
            bytemuck::cast_slice(&[UniformBuffer {
                screen_size: [logical_width as f32, logical_height as f32],
                _padding: [0.0; 2],
            }]),
        );

        for (i, egui::ClippedMesh(_, mesh)) in paint_jobs.iter().enumerate() {
            let data: &[u8] = bytemuck::cast_slice(&mesh.indices);
            if i < index_size {
                self.update_buffer(device, queue, BufferType::Index, i, data)
            } else {
                let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("egui_index_buffer"),
                    contents: data,
                    usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                });
                self.index_buffers.push(SizedBuffer {
                    buffer,
                    size: data.len(),
                });
            }

            let data: &[u8] = bytemuck::cast_slice(&mesh.vertices);
            if i < vertex_size {
                self.update_buffer(device, queue, BufferType::Vertex, i, data)
            } else {
                let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("egui_vertex_buffer"),
                    contents: data,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                });

                self.vertex_buffers.push(SizedBuffer {
                    buffer,
                    size: data.len(),
                });
            }
        }
    }

    /// Updates the buffers used by egui. Will properly re-size the buffers if needed.
    fn update_buffer(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer_type: BufferType,
        index: usize,
        data: &[u8],
    ) {
        let (buffer, storage, name) = match buffer_type {
            BufferType::Index => (
                &mut self.index_buffers[index],
                wgpu::BufferUsages::INDEX,
                "index",
            ),
            BufferType::Vertex => (
                &mut self.vertex_buffers[index],
                wgpu::BufferUsages::VERTEX,
                "vertex",
            ),
            BufferType::Uniform => (
                &mut self.uniform_buffer,
                wgpu::BufferUsages::UNIFORM,
                "uniform",
            ),
        };

        if data.len() > buffer.size {
            buffer.size = data.len();
            buffer.buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(format!("egui_{}_buffer", name).as_str()),
                contents: bytemuck::cast_slice(data),
                usage: storage | wgpu::BufferUsages::COPY_DST,
            });
        } else {
            queue.write_buffer(&buffer.buffer, 0, data);
        }
    }
}

/// Create a texture and bind group from existing data
fn create_texture_and_bind_group(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label_base: &str,
    origin: wgpu::Origin3d,
    image_data: &[u8],
    image_data_layout: wgpu::ImageDataLayout,
    image_size: wgpu::Extent3d,
    texture_bind_group_layout: &wgpu::BindGroupLayout,
) -> (wgpu::Texture, wgpu::BindGroup) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(format!("{}_texture", label_base).as_str()),
        size: image_size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    });

    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin,
            aspect: wgpu::TextureAspect::All,
        },
        image_data,
        image_data_layout,
        image_size,
    );

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some(format!("{}_sampler", label_base).as_str()),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(format!("{}_texture_bind_group", label_base).as_str()),
        layout: &texture_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(
                    &texture.create_view(&wgpu::TextureViewDescriptor::default()),
                ),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });

    (texture, bind_group)
}
