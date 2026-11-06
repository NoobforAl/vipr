//! Log mel filter bank features for the WeSpeaker embedding model (Kaldi style, 80 bins, 16 kHz).
//!
//! The model was trained on exactly these numbers, so every setting and step below follows Kaldi: 25 ms frames every
//! 10 ms that must fit in the audio, DC removal, 0.97 pre-emphasis, Povey window, 512-point FFT, triangular mel bins
//! from 20 Hz to Nyquist, log of the power with `f32::EPSILON` as floor, no dither. Each bin's mean over the whole
//! clip is then subtracted. Single precision is deliberate, it is what the reference implementation computes in.

use anyhow::{Result, bail};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;

/// Number of values per frame.
pub const NUM_BINS: usize = 80;

const SAMPLE_RATE: f32 = 16_000.0;
const FRAME_LENGTH: usize = 400;
const FRAME_SHIFT: usize = 160;
const FFT_SIZE: usize = 512;
const LOW_FREQ: f32 = 20.0;
const PREEMPHASIS: f32 = 0.97;

fn mel_scale(freq: f32) -> f32 {
    1127.0 * (1.0 + freq / 700.0).ln()
}

fn povey_window() -> Vec<f32> {
    let a = std::f64::consts::TAU / (FRAME_LENGTH - 1) as f64;
    (0..FRAME_LENGTH)
        .map(|i| (0.5 - 0.5 * (a * i as f64).cos()).powf(0.85) as f32)
        .collect()
}

/// For each mel bin: the first FFT bin it covers and the weights from there on.
fn mel_banks() -> Vec<(usize, Vec<f32>)> {
    let bin_width = SAMPLE_RATE / FFT_SIZE as f32;
    let mel_low = mel_scale(LOW_FREQ);
    let mel_high = mel_scale(0.5 * SAMPLE_RATE);
    let delta = (mel_high - mel_low) / (NUM_BINS + 1) as f32;
    (0..NUM_BINS)
        .map(|bin| {
            let left = mel_low + bin as f32 * delta;
            let center = mel_low + (bin + 1) as f32 * delta;
            let right = mel_low + (bin + 2) as f32 * delta;
            let mut first = None;
            let mut weights = Vec::new();
            for i in 0..FFT_SIZE / 2 {
                let mel = mel_scale(bin_width * i as f32);
                if mel > left && mel < right {
                    first.get_or_insert(i);
                    weights.push(if mel <= center {
                        (mel - left) / (center - left)
                    } else {
                        (right - mel) / (right - center)
                    });
                }
            }
            (first.unwrap_or(0), weights)
        })
        .collect()
}

/// Features of a clip: the number of frames and the values, frame after frame, [`NUM_BINS`] per frame.
pub fn compute_fbank(samples: &[f32]) -> Result<(usize, Vec<f32>)> {
    if samples.len() < FRAME_LENGTH {
        bail!(
            "{} samples are too few for one frame ({FRAME_LENGTH})",
            samples.len()
        );
    }
    let frames = 1 + (samples.len() - FRAME_LENGTH) / FRAME_SHIFT;
    let window = povey_window();
    let banks = mel_banks();
    let fft = FftPlanner::<f32>::new().plan_fft_forward(FFT_SIZE);
    let mut scratch = vec![Complex::default(); fft.get_inplace_scratch_len()];
    let mut buffer = vec![Complex::<f32>::default(); FFT_SIZE];
    let mut frame = [0f32; FRAME_LENGTH];
    let mut power = [0f32; FFT_SIZE / 2];
    let mut out = Vec::with_capacity(frames * NUM_BINS);

    for f in 0..frames {
        frame.copy_from_slice(&samples[f * FRAME_SHIFT..f * FRAME_SHIFT + FRAME_LENGTH]);
        let mean = frame.iter().sum::<f32>() / FRAME_LENGTH as f32;
        frame.iter_mut().for_each(|s| *s -= mean);
        for i in (1..FRAME_LENGTH).rev() {
            frame[i] -= PREEMPHASIS * frame[i - 1];
        }
        frame[0] -= PREEMPHASIS * frame[0];

        buffer.fill(Complex::default());
        for (b, (s, w)) in buffer.iter_mut().zip(frame.iter().zip(&window)) {
            b.re = s * w;
        }
        fft.process_with_scratch(&mut buffer, &mut scratch);
        for (p, c) in power.iter_mut().zip(&buffer) {
            *p = c.re * c.re + c.im * c.im;
        }

        for (first, weights) in &banks {
            let energy: f32 = weights
                .iter()
                .zip(&power[*first..])
                .map(|(w, p)| w * p)
                .sum();
            out.push(energy.max(f32::EPSILON).ln());
        }
    }

    for bin in 0..NUM_BINS {
        let mean = out.iter().skip(bin).step_by(NUM_BINS).sum::<f32>() / frames as f32;
        out.iter_mut()
            .skip(bin)
            .step_by(NUM_BINS)
            .for_each(|v| *v -= mean);
    }
    Ok((frames, out))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Sines plus repeatable noise, one second long.
    fn test_signal() -> Vec<f32> {
        let mut noise = 0x1234_5678u32;
        (0..16_000)
            .map(|i| {
                noise = noise.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let t = i as f32 / 16_000.0;
                0.3 * (std::f32::consts::TAU * 440.0 * t).sin()
                    + 0.1 * (std::f32::consts::TAU * 3_000.0 * t).sin()
                    + 0.05 * ((noise >> 16) as f32 / 32_768.0 - 1.0)
            })
            .collect()
    }

    /// Reference values computed by kaldi-native-fbank (through knf-rs 0.3.2) before it was replaced.
    #[test]
    fn matches_kaldi_reference() {
        let (frames, got) = compute_fbank(&test_signal()).unwrap();
        assert_eq!(frames, 98);
        assert_eq!(got.len(), 98 * NUM_BINS);
        for (f, b, want) in [
            (0, 0, 0.89408875f32),
            (0, 10, -1.0133986),
            (0, 40, -0.6848979),
            (0, 79, -0.017767072),
            (50, 0, -0.36037254),
            (50, 10, 0.8250289),
            (50, 40, 1.4361821),
            (50, 79, -0.4087271),
            (97, 0, 0.6326666),
            (97, 10, 1.0694718),
            (97, 40, 0.022563696),
            (97, 79, 0.10256624),
        ] {
            let v = got[f * NUM_BINS + b];
            assert!((v - want).abs() < 1e-3, "frame {f} bin {b}: {v} vs {want}");
        }
    }

    #[test]
    fn too_short_is_an_error() {
        assert!(compute_fbank(&[]).is_err());
        assert!(compute_fbank(&[0.0; FRAME_LENGTH - 1]).is_err());
        assert_eq!(compute_fbank(&[0.0; FRAME_LENGTH]).unwrap().0, 1);
    }
}
