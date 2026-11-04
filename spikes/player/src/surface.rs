//! The video surface: an `iced::widget::shader` program using Iced's own device and queue.

use crate::session::{Frame, Shared};
use iced::mouse;
use iced::time::Instant;
use iced::wgpu;
use iced::widget::shader::{self, Action};
use iced::window;
use iced::{Event, Rectangle};
use std::sync::Arc;
use std::sync::atomic::Ordering::*;
use std::time::Duration;

const WGSL: &str = r#"
struct Uniforms { scale: vec2<f32>, srgb_target: u32, _pad: u32 };
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var y_tex: texture_2d<f32>;
@group(0) @binding(2) var u_tex: texture_2d<f32>;
@group(0) @binding(3) var v_tex: texture_2d<f32>;
@group(0) @binding(4) var samp: sampler;

struct VsOut { @builtin(position) pos: vec4<f32>, @location(0) ndc: vec2<f32> };

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    var p = array<vec2<f32>, 4>(vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, 1.0));
    var o: VsOut;
    o.pos = vec4(p[i], 0.0, 1.0);
    o.ndc = p[i];
    return o;
}

fn to_linear(c: vec3<f32>) -> vec3<f32> {
    let low = c / 12.92;
    let high = pow((c + 0.055) / 1.055, vec3(2.4));
    return select(high, low, c <= vec3(0.04045));
}

@fragment
fn fs(in: VsOut) -> @location(0) vec4<f32> {
    let q = in.ndc / u.scale;
    if (abs(q.x) > 1.0 || abs(q.y) > 1.0) {
        return vec4(0.0, 0.0, 0.0, 1.0); // letterbox
    }
    let uv = vec2(q.x * 0.5 + 0.5, 0.5 - q.y * 0.5);
    // BT.709 limited range
    let yy = (textureSample(y_tex, samp, uv).r - 16.0 / 255.0) * (255.0 / 219.0);
    let cb = (textureSample(u_tex, samp, uv).r - 128.0 / 255.0) * (255.0 / 224.0);
    let cr = (textureSample(v_tex, samp, uv).r - 128.0 / 255.0) * (255.0 / 224.0);
    var rgb = clamp(vec3(
        yy + 1.5748 * cr,
        yy - 0.1873 * cb - 0.4681 * cr,
        yy + 1.8556 * cb,
    ), vec3(0.0), vec3(1.0));
    if (u.srgb_target != 0u) {
        // an sRGB target encodes on write; decode so that what reaches the screen equals the video's values
        rgb = to_linear(rgb);
    }
    return vec4(rgb, 1.0);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    scale: [f32; 2],
    srgb_target: u32,
    _pad: u32,
}

pub struct VideoPipeline {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    uniforms: wgpu::Buffer,
    srgb_target: bool,
    size: (u32, u32),
    textures: Option<[wgpu::Texture; 3]>,
    bind_group: Option<wgpu::BindGroup>,
    last_seq: u64,
    last_scale: [f32; 2],
    stats_hooked: bool,
    format: wgpu::TextureFormat,
    device: wgpu::Device,
}

impl shader::Pipeline for VideoPipeline {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("video"),
            source: wgpu::ShaderSource::Wgsl(WGSL.into()),
        });
        let texture_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("video"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                texture_entry(1),
                texture_entry(2),
                texture_entry(3),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("video"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("video"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("video"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("video"),
            size: size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut this = Self {
            pipeline,
            layout,
            sampler,
            uniforms,
            srgb_target: format.is_srgb(),
            size: (0, 0),
            textures: None,
            bind_group: None,
            last_seq: 0,
            last_scale: [1.0, 1.0],
            stats_hooked: false,
            format,
            device: device.clone(),
        };
        // Until the first real frame the surface shows video black, 1x1 Y=16 U=V=128, instead of leaving the
        // widget undrawn and the window's theme background (white in a light theme) visible through it.
        // All-zero planes would not do: in YUV they decode to green.
        this.ensure_textures(1, 1);
        if let Some(t) = this.textures.as_ref() {
            write_plane(queue, &t[0], &[16], 1, 1);
            write_plane(queue, &t[1], &[128], 1, 1);
            write_plane(queue, &t[2], &[128], 1, 1);
        }
        let u = Uniforms { scale: [1.0, 1.0], srgb_target: this.srgb_target as u32, _pad: 0 };
        queue.write_buffer(&this.uniforms, 0, bytemuck::bytes_of(&u));
        this
    }
}

impl VideoPipeline {
    fn ensure_textures(&mut self, w: u32, h: u32) {
        if self.size == (w, h) && self.textures.is_some() {
            return;
        }
        let make = |w, h| {
            self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("video plane"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };
        let (cw, ch) = ((w / 2).max(1), (h / 2).max(1));
        let textures = [make(w, h), make(cw, ch), make(cw, ch)];
        let view = |t: &wgpu::Texture| t.create_view(&wgpu::TextureViewDescriptor::default());
        let (v0, v1, v2) = (view(&textures[0]), view(&textures[1]), view(&textures[2]));
        self.bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("video"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.uniforms.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&v0) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&v1) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&v2) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        }));
        self.textures = Some(textures);
        self.size = (w, h);
    }
}

fn write_plane(queue: &wgpu::Queue, tex: &wgpu::Texture, data: &[u8], w: u32, h: u32) {
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        data,
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w), rows_per_image: Some(h) },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
}

pub struct VideoPrimitive {
    frame: Arc<Frame>,
    shared: Arc<Shared>,
}

impl std::fmt::Debug for VideoPrimitive {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VideoPrimitive").field("seq", &self.frame.seq).finish()
    }
}

impl shader::Primitive for VideoPrimitive {
    type Pipeline = VideoPipeline;

    fn prepare(
        &self,
        pipeline: &mut VideoPipeline,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bounds: &Rectangle,
        _viewport: &shader::Viewport,
    ) {
        let stats = &self.shared.stats;
        if !pipeline.stats_hooked {
            pipeline.stats_hooked = true;
            self.shared.surface_ready.store(true, Release);
            if let Ok(mut g) = stats.surface_format.lock() {
                *g = format!("{:?} (sRGB decode in shader: {})", pipeline.format, pipeline.srgb_target);
            }
            let hook = stats.clone();
            device.set_device_lost_callback(move |reason, message| {
                hook.c.device_lost.fetch_add(1, Relaxed);
                if let Ok(mut g) = hook.device_lost_log.lock() {
                    g.push(format!("{reason:?}: {message}"));
                }
                log::error!("device lost: {reason:?}: {message}");
            });
        }
        let f = &self.frame;
        if f.width == 0 {
            return;
        }
        let new_size = pipeline.size != (f.width, f.height);
        pipeline.ensure_textures(f.width, f.height);

        // letterbox, in clip space of the widget's bounds
        let (va, ba) = (f.width as f32 / f.height as f32, bounds.width / bounds.height.max(1.0));
        let scale = if ba > va { [va / ba, 1.0] } else { [1.0, ba / va] };
        if new_size || scale != pipeline.last_scale {
            pipeline.last_scale = scale;
            let u = Uniforms { scale, srgb_target: pipeline.srgb_target as u32, _pad: 0 };
            queue.write_buffer(&pipeline.uniforms, 0, bytemuck::bytes_of(&u));
        }

        if f.seq != pipeline.last_seq {
            pipeline.last_seq = f.seq;
            let began = std::time::Instant::now();
            let (y, u, v) = f.planes();
            let t = pipeline.textures.as_ref().expect("textures");
            write_plane(queue, &t[0], y, f.width, f.height);
            write_plane(queue, &t[1], u, f.width / 2, f.height / 2);
            write_plane(queue, &t[2], v, f.width / 2, f.height / 2);
            let cost = began.elapsed().as_secs_f32() * 1000.0;
            let err = (self.shared.clock.now() - f.pts) as f32 * 1000.0;
            stats.record_upload(err, cost);
            stats.seek_first_upload(f.generation);
        }
    }

    fn draw(&self, pipeline: &VideoPipeline, pass: &mut wgpu::RenderPass<'_>) -> bool {
        let Some(group) = pipeline.bind_group.as_ref() else {
            return true;
        };
        pass.set_pipeline(&pipeline.pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.draw(0..4, 0..1);
        true
    }
}

pub struct VideoProgram {
    pub shared: Arc<Shared>,
}

impl<Message> shader::Program<Message> for VideoProgram {
    type State = ();
    type Primitive = VideoPrimitive;

    /// Redraws are chained here, so playback causes no Message and no `view()` call per frame.
    fn update(
        &self,
        _state: &mut (),
        event: &Event,
        _bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Option<Action<Message>> {
        if !matches!(event, Event::Window(window::Event::RedrawRequested(_))) {
            return None;
        }
        let shared = &self.shared;
        shared.stats.c.redraws.fetch_add(1, Relaxed);
        if shared.paused.load(Acquire) || shared.ended.load(Acquire) {
            return None;
        }
        let interval = shared.frame_interval;
        let shown = shared.latest.lock().map(|g| g.pts).unwrap_or(0.0);
        // wake just after the next frame is due on the clock
        let wait = (shown + interval - shared.clock.now() + 0.001).clamp(0.001, interval);
        Some(Action::request_redraw_at(Instant::now() + Duration::from_secs_f64(wait)))
    }

    fn draw(&self, _state: &(), _cursor: mouse::Cursor, _bounds: Rectangle) -> VideoPrimitive {
        let frame = self
            .shared
            .latest
            .lock()
            .map(|g| g.clone())
            .unwrap_or_else(|_| Arc::new(Frame::empty()));
        VideoPrimitive { frame, shared: self.shared.clone() }
    }
}
