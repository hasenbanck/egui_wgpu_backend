use bytemuck::{Pod, Zeroable};
use wgpu::util::{BufferInitDescriptor, DeviceExt};

/// Enum for selecting the right buffer type.
#[derive(Debug)]
enum BufferType {
    Vertex,
    Index,
}

/// PushConstants used when rendering.
#[derive(Clone, Copy, Debug)]
struct EguiPushConstants {
    screen_size: [f32; 2],
}

unsafe impl Pod for EguiPushConstants {}
unsafe impl Zeroable for EguiPushConstants {}

/// Wraps the buffers and includes additional information.
#[derive(Debug)]
struct SizedBuffer {
    buffer: wgpu::Buffer,
    size: usize,
}

/// RenderPass to render a egui based GUI.
pub struct EguiRenderPass {
    render_pipeline: wgpu::RenderPipeline,
    index_buffers: Vec<SizedBuffer>,
    vertex_buffers: Vec<SizedBuffer>,
    texture_bind_group_layout: wgpu::BindGroupLayout,
    texture_bind_group: Option<wgpu::BindGroup>,
    texture_version: Option<u64>,
    texture_width: u32,
    texture_height: u32,
}

impl EguiRenderPass {
    /// Creates a new egui render pass.
    pub fn new(device: &wgpu::Device, output_format: wgpu::TextureFormat) -> Self {
        let vs_module = device.create_shader_module(wgpu::util::make_spirv(include_bytes!(
            r#"../gen/shader/egui.vert.spv"#
        )));
        let fs_module = device.create_shader_module(wgpu::util::make_spirv(include_bytes!(
            r#"../gen/shader/egui.frag.spv"#
        )));

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("egui_texture_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::SampledTexture {
                        multisampled: false,
                        component_type: wgpu::TextureComponentType::Float,
                        dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Sampler { comparison: false },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("egui_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStage::VERTEX,
                range: (0..8),
            }],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("egui_pipeline"),
            layout: Some(&pipeline_layout),
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &vs_module,
                entry_point: "main",
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &fs_module,
                entry_point: "main",
            }),
            rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: wgpu::CullMode::None,
                clamp_depth: false,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0,
                polygon_mode: wgpu::PolygonMode::Fill,
            }),
            primitive_topology: wgpu::PrimitiveTopology::TriangleList,
            color_states: &[wgpu::ColorStateDescriptor {
                format: output_format,
                color_blend: wgpu::BlendDescriptor {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha_blend: wgpu::BlendDescriptor {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
                write_mask: wgpu::ColorWrite::ALL,
            }],
            depth_stencil_state: None,
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: wgpu::IndexFormat::Uint32,
                vertex_buffers: &[wgpu::VertexBufferDescriptor {
                    stride: 5 * 4,
                    step_mode: wgpu::InputStepMode::Vertex,
                    // 0: vec2 position
                    // 1: vec2 texture coordinates
                    // 2: uint color
                    attributes: &wgpu::vertex_attr_array![0 => Float2, 1 => Float2, 2 => Uint],
                }],
            },
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        Self {
            render_pipeline,
            vertex_buffers: Vec::with_capacity(64),
            index_buffers: Vec::with_capacity(64),
            texture_bind_group_layout: bind_group_layout,
            texture_version: None,
            texture_bind_group: None,
            texture_width: 0,
            texture_height: 0,
        }
    }

    /// Executes the egui render pass.
    pub fn execute(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        color_attachment: &wgpu::TextureView,
        paint_jobs: &egui::PaintJobs,
        physical_width: u32,
        physical_height: u32,
        scale_factor: f32,
        clear_color: Option<wgpu::Color>
    ) {
        let load_operation = if let Some(color) = clear_color {
            wgpu::LoadOp::Clear(color)
        } else {
            wgpu::LoadOp::Load
        };

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[wgpu::RenderPassColorAttachmentDescriptor {
                attachment: color_attachment,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: load_operation,
                    store: true,
                },
            }],
            depth_stencil_attachment: None,
        });
        pass.push_debug_group("egui_pass");
        pass.set_pipeline(&self.render_pipeline);

        let logical_width = physical_width as f32 / scale_factor;
        let logical_height = physical_height as f32 / scale_factor;

        pass.set_push_constants(
            wgpu::ShaderStage::VERTEX,
            0,
            bytemuck::cast_slice(&[EguiPushConstants {
                screen_size: [logical_width as f32, logical_height as f32],
            }]),
        );

        for (((clip_rect, triangles), vertex_buffer), index_buffer) in paint_jobs
            .iter()
            .zip(self.vertex_buffers.iter())
            .zip(self.index_buffers.iter())
        {
            pass.set_index_buffer(index_buffer.buffer.slice(..));
            pass.set_vertex_buffer(0, vertex_buffer.buffer.slice(..));
            pass.set_bind_group(
                0,
                self.texture_bind_group
                    .as_ref()
                    .unwrap_or_else(|| panic!("egui texture was not set before the first draw")),
                &[],
            );

            // Transform clip rect to physical pixels.
            let clip_min_x = scale_factor * clip_rect.min.x;
            let clip_min_y = scale_factor * clip_rect.min.y;
            let clip_max_x = scale_factor * clip_rect.max.x;
            let clip_max_y = scale_factor * clip_rect.max.y;

            // Make sure clip rect can fit withing an `u32`.
            let clip_min_x = egui::clamp(clip_min_x, 0.0..=physical_width as f32);
            let clip_min_y = egui::clamp(clip_min_y, 0.0..=physical_height as f32);
            let clip_max_x = egui::clamp(clip_max_x, clip_min_x..=physical_width as f32);
            let clip_max_y = egui::clamp(clip_max_y, clip_min_y..=physical_height as f32);

            let clip_min_x = clip_min_x.round() as u32;
            let clip_min_y = clip_min_y.round() as u32;
            let clip_max_x = clip_max_x.round() as u32;
            let clip_max_y = clip_max_y.round() as u32;

            pass.set_scissor_rect(
                clip_min_x,
                clip_min_y,
                clip_max_x - clip_min_x,
                clip_max_y - clip_min_y,
            );

            pass.draw_indexed(0..triangles.indices.len() as u32, 0, 0..1);
        }

        pass.pop_debug_group();
    }

    /// Updates the texture used by egui for the fonts etc. When `clear_on_draw` is set, the output target will get cleared before writing to it.
    pub fn update_texture(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        egui_texture: &egui::Texture,
    ) {
        // Don't update the font texture if it hasn't changed.
        if self.texture_version == Some(egui_texture.version) {
            return;
        }

        let size = wgpu::Extent3d {
            width: egui_texture.width as u32,
            height: egui_texture.height as u32,
            depth: 1,
        };

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("egui_font_texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST,
        });

        queue.write_texture(
            wgpu::TextureCopyView {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
            },
            egui_texture.pixels.as_slice(),
            wgpu::TextureDataLayout {
                offset: 0,
                bytes_per_row: (egui_texture.pixels.len() / egui_texture.height) as u32,
                rows_per_image: egui_texture.height as u32,
            },
            size,
        );

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("egui_texture"),
            ..Default::default()
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("egui_font_texture_bind_group"),
            layout: &self.texture_bind_group_layout,
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

        self.texture_width = egui_texture.width as u32;
        self.texture_height = egui_texture.height as u32;
        self.texture_version = Some(egui_texture.version);
        self.texture_bind_group = Some(bind_group);
    }

    /// Uploads the vertex and index data used by the imgui render pass.
    pub fn update_buffers(
        &mut self,
        device: &mut wgpu::Device,
        queue: &mut wgpu::Queue,
        paint_jobs: &egui::PaintJobs,
    ) {
        let index_size = self.index_buffers.len();
        let vertex_size = self.vertex_buffers.len();

        for (i, (_, triangles)) in paint_jobs.iter().enumerate() {
            let data: &[u8] = bytemuck::cast_slice(&triangles.indices);
            if i < index_size {
                self.update_buffer(device, queue, BufferType::Index, i, data)
            } else {
                let buffer = device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("egui_index_buffer"),
                    contents: data,
                    usage: wgpu::BufferUsage::INDEX | wgpu::BufferUsage::COPY_DST,
                });
                self.index_buffers.push(SizedBuffer {
                    buffer,
                    size: data.len(),
                });
            }

            let data: &[u8] = as_byte_slice(&triangles.vertices);
            if i < vertex_size {
                self.update_buffer(device, queue, BufferType::Vertex, i, data)
            } else {
                let buffer = device.create_buffer_init(&BufferInitDescriptor {
                    label: Some("egui_vertex_buffer"),
                    contents: data,
                    usage: wgpu::BufferUsage::VERTEX | wgpu::BufferUsage::COPY_DST,
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
        device: &mut wgpu::Device,
        queue: &mut wgpu::Queue,
        buffer_type: BufferType,
        index: usize,
        data: &[u8],
    ) {
        let (buffer, storage, name) = match buffer_type {
            BufferType::Index => (
                &mut self.index_buffers[index],
                wgpu::BufferUsage::INDEX,
                "index",
            ),
            BufferType::Vertex => (
                &mut self.vertex_buffers[index],
                wgpu::BufferUsage::VERTEX,
                "vertex",
            ),
        };

        if data.len() > buffer.size {
            buffer.size = data.len();
            buffer.buffer = device.create_buffer_init(&BufferInitDescriptor {
                label: Some(format!("egui_{}_buffer", name).as_str()),
                contents: bytemuck::cast_slice(data),
                usage: storage | wgpu::BufferUsage::COPY_DST,
            });
        } else {
            queue.write_buffer(&buffer.buffer, 0, data);
        }
    }
}

// Needed since we can't use bytemuck for external types.
fn as_byte_slice<T>(slice: &[T]) -> &[u8] {
    let len = slice.len() * std::mem::size_of::<T>();
    let ptr = slice.as_ptr() as *const u8;
    unsafe { std::slice::from_raw_parts(ptr, len) }
}
