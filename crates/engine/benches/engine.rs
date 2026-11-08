//! Benchmarks for the engine's hot paths: resampling decoded audio and reading subtitle files.
//! Run with `cargo bench -p engine` (or `make bench`).

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use engine::audio::{MonoResampler, resample_mono};

/// `secs` seconds of a tone with some movement in it, at `rate` Hz.
fn clip(rate: u32, secs: usize) -> Vec<f32> {
    (0..rate as usize * secs)
        .map(|i| (i as f32 * 0.0577).sin() * 0.4 + (i as f32 * 0.0013).sin() * 0.1)
        .collect()
}

fn resample(c: &mut Criterion) {
    let mut g = c.benchmark_group("resample_to_16k");
    let secs = 60;
    for rate in [48_000u32, 44_100] {
        let input = clip(rate, secs);
        g.throughput(Throughput::Elements(input.len() as u64));
        // as decoding feeds it: one packet (1024 frames, AAC's size) at a time
        g.bench_with_input(BenchmarkId::new("streamed", rate), &input, |b, input| {
            b.iter(|| {
                let mut rs = MonoResampler::new(rate, 16_000, input.len()).unwrap();
                for packet in input.chunks(1024) {
                    rs.push(packet).unwrap();
                }
                black_box(rs.finish().unwrap())
            })
        });
        g.bench_with_input(BenchmarkId::new("whole", rate), &input, |b, input| {
            b.iter(|| black_box(resample_mono(input.clone(), rate, 16_000).unwrap()))
        });
    }
    g.finish();
}

fn subtitles(c: &mut Criterion) {
    // a two-hour meeting's worth of cues
    let mut srt = String::new();
    for i in 0..2400u32 {
        let (a, b) = (i * 3, i * 3 + 2);
        srt.push_str(&format!(
            "{}\n{:02}:{:02}:{:02},000 --> {:02}:{:02}:{:02},500\nSpeaker {}: line {i} of what was said,\nand a second line.\n\n",
            i + 1,
            a / 3600,
            a / 60 % 60,
            a % 60,
            b / 3600,
            b / 60 % 60,
            b % 60,
            i % 4
        ));
    }
    let bytes = srt.as_bytes().to_vec();
    let mut g = c.benchmark_group("subtitles");
    g.throughput(Throughput::Bytes(bytes.len() as u64));
    g.bench_function("decode_and_parse_srt", |b| {
        b.iter(|| {
            let text = engine::subparse::decode(black_box(&bytes));
            black_box(engine::subparse::parse(&text.text))
        })
    });
    g.finish();
}

criterion_group!(benches, resample, subtitles);
criterion_main!(benches);
