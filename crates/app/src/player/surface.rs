//! The video surface: an `iced::widget::shader` program on Iced's own device and queue.

use std::sync::Arc;
use std::sync::atomic::Ordering::Release;
use std::time::Duration;

use iced::time::Instant;
use iced::wgpu;
use iced::widget::shader::{self, Action};
use iced::{Event, Rectangle, mouse, window};

use super::session::{Frame, Shared};

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

/// The WGSL `Uniforms`: scale.x, scale.y, srgb_target, padding.
fn uniform_bytes(scale: [f32; 2], srgb_target: bool) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[0..4].copy_from_slice(&scale[0].to_ne_bytes());
    out[4..8].copy_from_slice(&scale[1].to_ne_bytes());
    out[8..12].copy_from_slice(&(srgb_target as u32).to_ne_bytes());
    out
}

/// How much of the widget's clip space the picture covers, in proportion, with bars on the rest.
fn letterbox(video: (u32, u32), bounds: (f32, f32)) -> [f32; 2] {
    let va = video.0 as f32 / video.1.max(1) as f32;
    let ba = bounds.0 / bounds.1.max(1.0);
    if ba > va {
        [va / ba, 1.0]
    } else {
        [1.0, ba / va]
    }
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
    /// Zero while showing black.
    last_seq: u64,
    last_scale: [f32; 2],
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
            size: 16,
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
            device: device.clone(),
        };
        this.write_black(queue);
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
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
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
        let plane = |binding, view| wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::TextureView(view),
        };
        self.bind_group = Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("video"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniforms.as_entire_binding(),
                },
                plane(1, &v0),
                plane(2, &v1),
                plane(3, &v2),
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        }));
        self.textures = Some(textures);
        self.size = (w, h);
    }

    /// Until a session's first real frame the surface shows video black, 1x1 Y=16 U=V=128, instead
    /// of the previous session's last picture or the window's theme background (white in a light
    /// theme). All-zero planes would not do: in YUV they decode to green.
    fn write_black(&mut self, queue: &wgpu::Queue) {
        self.ensure_textures(1, 1);
        if let Some(t) = self.textures.as_ref() {
            write_plane(queue, &t[0], &[16], 1, 1);
            write_plane(queue, &t[1], &[128], 1, 1);
            write_plane(queue, &t[2], &[128], 1, 1);
        }
        self.last_scale = [1.0, 1.0];
        self.last_seq = 0;
        let bytes = uniform_bytes(self.last_scale, self.srgb_target);
        queue.write_buffer(&self.uniforms, 0, &bytes);
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
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(w),
            rows_per_image: Some(h),
        },
        wgpu::Extent3d {
            width: w,
            height: h,
            depth_or_array_layers: 1,
        },
    );
}

pub struct VideoPrimitive {
    frame: Arc<Frame>,
    shared: Arc<Shared>,
}

impl std::fmt::Debug for VideoPrimitive {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VideoPrimitive")
            .field("seq", &self.frame.seq)
            .finish()
    }
}

impl shader::Primitive for VideoPrimitive {
    type Pipeline = VideoPipeline;

    fn prepare(
        &self,
        pipeline: &mut VideoPipeline,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        bounds: &Rectangle,
        _viewport: &shader::Viewport,
    ) {
        // what the video thread waits for before it lets the clock and the sound go
        self.shared.surface_ready.store(true, Release);

        let f = &self.frame;
        if f.width == 0 {
            // Iced keeps one pipeline for the app, so this may still hold the last session
            if pipeline.last_seq != 0 {
                pipeline.write_black(queue);
            }
            return;
        }
        let new_size = pipeline.size != (f.width, f.height);
        pipeline.ensure_textures(f.width, f.height);

        let scale = letterbox((f.width, f.height), (bounds.width, bounds.height));
        if new_size || scale != pipeline.last_scale {
            pipeline.last_scale = scale;
            let bytes = uniform_bytes(scale, pipeline.srgb_target);
            queue.write_buffer(&pipeline.uniforms, 0, &bytes);
        }

        if f.seq != pipeline.last_seq {
            pipeline.last_seq = f.seq;
            let (y, u, v) = f.planes();
            if let Some(t) = pipeline.textures.as_ref() {
                let (cw, ch) = (f.width / 2, f.height / 2);
                write_plane(queue, &t[0], y, f.width, f.height);
                write_plane(queue, &t[1], u, cw, ch);
                write_plane(queue, &t[2], v, cw, ch);
            }
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

/// Whether a redraw chain can stop: nothing will change on screen until a message wakes it.
fn is_still(paused_or_ended: bool, seeking: bool, already_drawn: bool) -> bool {
    paused_or_ended && !seeking && already_drawn
}

pub struct VideoProgram {
    pub shared: Arc<Shared>,
}

impl<Message> shader::Program<Message> for VideoProgram {
    /// The frame of the last redraw.
    type State = u64;
    type Primitive = VideoPrimitive;

    /// Redraws are chained here, so playback causes no Message and no `view()` call per frame.
    fn update(
        &self,
        drawn: &mut u64,
        event: &Event,
        _bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Option<Action<Message>> {
        if !matches!(event, Event::Window(window::Event::RedrawRequested(_))) {
            return None;
        }
        let shared = &self.shared;
        let frame = shared.latest();
        let already_drawn = *drawn == frame.seq;
        *drawn = frame.seq;
        // a seek while paused still has to show its new picture, which arrives some time later
        let idle = shared.is_paused() || shared.ended();
        if is_still(idle, shared.is_seeking(), already_drawn) {
            return None;
        }
        let interval = shared.frame_interval;
        // wake just after the next frame is due on the clock
        let wait = (frame.pts + interval - shared.clock.now() + 0.001).clamp(0.001, interval);
        Some(Action::request_redraw_at(
            Instant::now() + Duration::from_secs_f64(wait),
        ))
    }

    fn draw(&self, _drawn: &u64, _cursor: mouse::Cursor, _bounds: Rectangle) -> VideoPrimitive {
        VideoPrimitive {
            frame: self.shared.latest(),
            shared: self.shared.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniforms_match_the_wgsl_layout() {
        let b = uniform_bytes([0.5, 1.0], true);
        assert_eq!(f32::from_ne_bytes(b[0..4].try_into().unwrap()), 0.5);
        assert_eq!(f32::from_ne_bytes(b[4..8].try_into().unwrap()), 1.0);
        assert_eq!(u32::from_ne_bytes(b[8..12].try_into().unwrap()), 1);
        assert_eq!(uniform_bytes([1.0, 1.0], false)[8..12], [0, 0, 0, 0]);
    }

    #[test]
    fn letterbox_keeps_proportions() {
        // wide window: bars left and right
        let [x, y] = letterbox((1600, 900), (1600.0, 450.0));
        assert!((x - 0.5).abs() < 1e-4 && y == 1.0);
        // tall window: bars above and below
        let [x, y] = letterbox((1600, 900), (800.0, 900.0));
        assert!(x == 1.0 && (y - 0.5).abs() < 1e-4);
        // exact fit
        assert_eq!(letterbox((1600, 900), (1600.0, 900.0)), [1.0, 1.0]);
        // no height yet must not divide by zero
        assert!(letterbox((1600, 900), (800.0, 0.0))[0].is_finite());
    }

    #[test]
    fn redraw_chain_stops_only_when_nothing_is_coming() {
        assert!(!is_still(false, false, true), "playing");
        assert!(is_still(true, false, true), "paused and drawn");
        assert!(!is_still(true, true, true), "seek on its way");
        assert!(!is_still(true, false, false), "new picture not drawn yet");
    }
}
