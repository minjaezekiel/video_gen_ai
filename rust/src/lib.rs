//! Native accelerator for `video_gen_ai`'s per-frame hot loops.
//!
//! This crate is **optional**. `video_gen.fastpath` falls back to NumPy and
//! Pillow when it is not installed, and the test suite checks the two
//! implementations agree pixel for pixel. Nothing here adds a capability; it
//! only makes an existing one cheaper.
//!
//! Three functions, each chosen because profiling a real render pointed at it:
//!
//! | measured | share of a 43.8 s render |
//! |---|---|
//! | `resize_crop` — Ken Burns crop + resize, once per frame | 20 % |
//! | `blend_box` — caption alpha composite, once per captioned frame | 4 % |
//! | `mouth_openness` — lip-sync envelope, once per shot | new work |
//!
//! Frames are passed as plain `bytes` rather than NumPy arrays so the whole
//! per-frame chain stays inside this crate without a copy at each hop: the
//! caller converts once, at the end, with `np.frombuffer`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use rayon::prelude::*;

const CHANNELS: usize = 3;

/// Pillow's `BILINEAR` is a triangle filter whose support widens when
/// downscaling, which is what keeps a shrunk image from aliasing. Matching it
/// exactly is the whole reason the fallback and the fast path can be compared.
#[inline]
fn triangle(x: f64) -> f64 {
    let x = x.abs();
    if x < 1.0 {
        1.0 - x
    } else {
        0.0
    }
}

/// Per-output-pixel filter taps, following Pillow's `precompute_coeffs`.
///
/// Returns `(bounds, coeffs, ksize)` where `bounds[i]` is
/// `(first source index, tap count)` for output pixel `i`, and `coeffs` is a
/// flat `out_size * ksize` table normalised so each row sums to 1.
fn precompute_coeffs(
    in_size: usize,
    in0: f64,
    in1: f64,
    out_size: usize,
) -> (Vec<(usize, usize)>, Vec<f64>, usize) {
    let scale = (in1 - in0) / out_size as f64;
    let filterscale = if scale < 1.0 { 1.0 } else { scale };
    let support = filterscale; // triangle support is 1.0, scaled
    let ksize = (support.ceil() as usize) * 2 + 1;

    let mut bounds = Vec::with_capacity(out_size);
    let mut coeffs = vec![0.0f64; out_size * ksize];
    let ss = 1.0 / filterscale;

    for xx in 0..out_size {
        let center = in0 + (xx as f64 + 0.5) * scale;
        let mut xmin = (center - support + 0.5).floor() as i64;
        if xmin < 0 {
            xmin = 0;
        }
        let mut xmax = (center + support + 0.5).floor() as i64;
        if xmax > in_size as i64 {
            xmax = in_size as i64;
        }
        let xmin = xmin as usize;
        let count = (xmax as usize).saturating_sub(xmin).min(ksize);

        let row = &mut coeffs[xx * ksize..xx * ksize + ksize];
        let mut total = 0.0;
        for i in 0..count {
            let w = triangle((i as f64 + xmin as f64 - center + 0.5) * ss);
            row[i] = w;
            total += w;
        }
        if total != 0.0 {
            for i in 0..count {
                row[i] /= total;
            }
        }
        bounds.push((xmin, count));
    }
    (bounds, coeffs, ksize)
}

#[inline]
fn clamp_u8(value: f64) -> u8 {
    // Pillow rounds half away from zero and clips; the +0.5 floor matches it.
    let v = value + 0.5;
    if v <= 0.0 {
        0
    } else if v >= 255.0 {
        255
    } else {
        v as u8
    }
}

/// Crop `src` to the given box and resample it to `out_w` x `out_h`.
///
/// Equivalent to `Image.resize((out_w, out_h), Image.BILINEAR, box=...)` on an
/// RGB image, which is exactly what the Ken Burns move does once per frame.
/// The crop is applied as an offset in the horizontal pass rather than by
/// copying the window out first.
#[allow(clippy::too_many_arguments)]
fn resize_crop_impl(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    x0: usize,
    y0: usize,
    crop_w: usize,
    crop_h: usize,
    out_w: usize,
    out_h: usize,
) -> Result<Vec<u8>, String> {
    if src.len() < src_w * src_h * CHANNELS {
        return Err(format!(
            "source buffer is {} bytes, need {} for {}x{} RGB",
            src.len(),
            src_w * src_h * CHANNELS,
            src_w,
            src_h
        ));
    }
    if crop_w == 0 || crop_h == 0 || out_w == 0 || out_h == 0 {
        return Err("crop and output dimensions must all be non-zero".into());
    }
    if x0 + crop_w > src_w || y0 + crop_h > src_h {
        return Err(format!(
            "crop box ({x0},{y0},{crop_w},{crop_h}) falls outside {src_w}x{src_h}"
        ));
    }

    // Pillow computes the vertical taps first, against the *full* source
    // height, so a filter window may reach outside the crop box and is clipped
    // only by the image itself. Clamping at the crop edge instead leaves a thin
    // band of wrong pixels along each side -- small in the mean, up to 80 in
    // the worst pixel, and the reason this mirrors Pillow's structure rather
    // than cropping first.
    let (mut v_bounds, v_coeffs, v_ksize) =
        precompute_coeffs(src_h, y0 as f64, (y0 + crop_h) as f64, out_h);

    let ybox_first = v_bounds.first().map(|b| b.0).unwrap_or(0);
    let ybox_last = v_bounds
        .iter()
        .map(|(start, count)| start + count)
        .max()
        .unwrap_or(0)
        .max(ybox_first);
    let band = ybox_last - ybox_first;
    if band == 0 {
        return Err("the crop box selects no source rows".into());
    }
    // Re-base the vertical taps onto the intermediate, which starts at
    // ybox_first rather than at row 0.
    for bound in v_bounds.iter_mut() {
        bound.0 -= ybox_first;
    }

    // Pass 1: horizontal, over just the band of rows the vertical pass needs.
    // The crop's x offset rides along in `in0`, so no window is copied out.
    let (h_bounds, h_coeffs, h_ksize) =
        precompute_coeffs(src_w, x0 as f64, (x0 + crop_w) as f64, out_w);

    let mut mid = vec![0u8; out_w * band * CHANNELS];
    mid.par_chunks_mut(out_w * CHANNELS)
        .enumerate()
        .for_each(|(row, out_row)| {
            let y = ybox_first + row;
            let src_row = &src[y * src_w * CHANNELS..(y + 1) * src_w * CHANNELS];
            for xx in 0..out_w {
                let (xmin, count) = h_bounds[xx];
                let k = &h_coeffs[xx * h_ksize..xx * h_ksize + h_ksize];
                let mut acc = [0.0f64; CHANNELS];
                for i in 0..count {
                    let p = (xmin + i) * CHANNELS;
                    let w = k[i];
                    acc[0] += src_row[p] as f64 * w;
                    acc[1] += src_row[p + 1] as f64 * w;
                    acc[2] += src_row[p + 2] as f64 * w;
                }
                let o = xx * CHANNELS;
                out_row[o] = clamp_u8(acc[0]);
                out_row[o + 1] = clamp_u8(acc[1]);
                out_row[o + 2] = clamp_u8(acc[2]);
            }
        });

    // Pass 2: vertical, over the intermediate band.
    let mut out = vec![0u8; out_w * out_h * CHANNELS];
    out.par_chunks_mut(out_w * CHANNELS)
        .enumerate()
        .for_each(|(yy, out_row)| {
            let (ymin, count) = v_bounds[yy];
            let k = &v_coeffs[yy * v_ksize..yy * v_ksize + v_ksize];
            for xx in 0..out_w {
                let mut acc = [0.0f64; CHANNELS];
                for i in 0..count {
                    let p = (ymin + i) * out_w * CHANNELS + xx * CHANNELS;
                    let w = k[i];
                    acc[0] += mid[p] as f64 * w;
                    acc[1] += mid[p + 1] as f64 * w;
                    acc[2] += mid[p + 2] as f64 * w;
                }
                let o = xx * CHANNELS;
                out_row[o] = clamp_u8(acc[0]);
                out_row[o + 1] = clamp_u8(acc[1]);
                out_row[o + 2] = clamp_u8(acc[2]);
            }
        });

    Ok(out)
}

/// Ken Burns: crop `src` and resample to the output size. RGB8 in, RGB8 out.
#[pyfunction]
#[pyo3(signature = (src, src_w, src_h, x0, y0, crop_w, crop_h, out_w, out_h))]
#[allow(clippy::too_many_arguments)]
fn resize_crop<'py>(
    py: Python<'py>,
    src: &[u8],
    src_w: usize,
    src_h: usize,
    x0: usize,
    y0: usize,
    crop_w: usize,
    crop_h: usize,
    out_w: usize,
    out_h: usize,
) -> PyResult<Bound<'py, PyBytes>> {
    let out = py
        .allow_threads(|| resize_crop_impl(src, src_w, src_h, x0, y0, crop_w, crop_h, out_w, out_h))
        .map_err(PyValueError::new_err)?;
    Ok(PyBytes::new_bound(py, &out))
}

/// Composite an RGBA box over an RGB frame, in place semantics, new buffer out.
///
/// Truncates rather than rounds, because the NumPy fallback ends in
/// `astype(np.uint8)` and the two have to agree exactly.
fn blend_box_impl(
    frame: &[u8],
    frame_w: usize,
    frame_h: usize,
    x0: usize,
    y0: usize,
    box_w: usize,
    box_h: usize,
    rgba: &[u8],
) -> Result<Vec<u8>, String> {
    if frame.len() < frame_w * frame_h * CHANNELS {
        return Err(format!(
            "frame buffer is {} bytes, need {} for {}x{} RGB",
            frame.len(),
            frame_w * frame_h * CHANNELS,
            frame_w,
            frame_h
        ));
    }
    if rgba.len() < box_w * box_h * 4 {
        return Err(format!(
            "overlay buffer is {} bytes, need {} for {}x{} RGBA",
            rgba.len(),
            box_w * box_h * 4,
            box_w,
            box_h
        ));
    }
    if x0 + box_w > frame_w || y0 + box_h > frame_h {
        return Err(format!(
            "overlay box ({x0},{y0},{box_w},{box_h}) falls outside {frame_w}x{frame_h}"
        ));
    }

    let mut out = frame.to_vec();
    out.par_chunks_mut(frame_w * CHANNELS)
        .enumerate()
        .for_each(|(row, out_row)| {
            if row < y0 || row >= y0 + box_h {
                return;
            }
            let src_row = (row - y0) * box_w * 4;
            for col in 0..box_w {
                let s = src_row + col * 4;
                let a = rgba[s + 3] as f32 / 255.0;
                if a <= 0.0 {
                    continue;
                }
                let d = (x0 + col) * CHANNELS;
                for c in 0..CHANNELS {
                    let blended = out_row[d + c] as f32 * (1.0 - a) + rgba[s + c] as f32 * a;
                    out_row[d + c] = blended as u8; // truncate, matching astype
                }
            }
        });
    Ok(out)
}

/// Alpha-composite an RGBA overlay onto an RGB frame.
#[pyfunction]
#[pyo3(signature = (frame, frame_w, frame_h, x0, y0, box_w, box_h, rgba))]
#[allow(clippy::too_many_arguments)]
fn blend_box<'py>(
    py: Python<'py>,
    frame: &[u8],
    frame_w: usize,
    frame_h: usize,
    x0: usize,
    y0: usize,
    box_w: usize,
    box_h: usize,
    rgba: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    let out = py
        .allow_threads(|| blend_box_impl(frame, frame_w, frame_h, x0, y0, box_w, box_h, rgba))
        .map_err(PyValueError::new_err)?;
    Ok(PyBytes::new_bound(py, &out))
}

/// Per-frame mouth openness in `[0, 1]` from a mono waveform.
///
/// A short-window RMS, normalised to the loudest window, then given an
/// asymmetric attack and release so a mouth snaps open and closes more slowly.
/// That asymmetry is what stops the result reading as a flicker.
fn mouth_openness_impl(
    samples: &[f32],
    sample_rate: f64,
    fps: f64,
    frames: usize,
    attack: f32,
    release: f32,
) -> Vec<f32> {
    if frames == 0 {
        return Vec::new();
    }
    if samples.is_empty() || sample_rate <= 0.0 || fps <= 0.0 {
        return vec![0.0; frames];
    }

    let per_frame = (sample_rate / fps).max(1.0);
    let mut raw: Vec<f32> = (0..frames)
        .into_par_iter()
        .map(|frame| {
            let start = (frame as f64 * per_frame) as usize;
            let end = (((frame + 1) as f64 * per_frame) as usize).min(samples.len());
            if start >= end {
                return 0.0;
            }
            let window = &samples[start..end];
            let sum: f64 = window.iter().map(|v| (*v as f64) * (*v as f64)).sum();
            (sum / window.len() as f64).sqrt() as f32
        })
        .collect();

    let peak = raw.iter().copied().fold(0.0f32, f32::max);
    if peak > 1e-6 {
        for value in raw.iter_mut() {
            // Square root compresses the range: speech RMS is mostly low, and
            // a linear map leaves the mouth barely moving.
            *value = (*value / peak).sqrt().clamp(0.0, 1.0);
        }
    }

    let mut smoothed = vec![0.0f32; frames];
    let mut level = 0.0f32;
    for (index, target) in raw.iter().enumerate() {
        let coeff = if *target > level { attack } else { release };
        level += (*target - level) * coeff.clamp(0.0, 1.0);
        smoothed[index] = level.clamp(0.0, 1.0);
    }
    smoothed
}

/// Mouth openness per frame, for the procedural tier's lip sync.
#[pyfunction]
#[pyo3(signature = (samples, sample_rate, fps, frames, attack = 0.65, release = 0.25))]
fn mouth_openness(
    py: Python<'_>,
    samples: Vec<f32>,
    sample_rate: f64,
    fps: f64,
    frames: usize,
    attack: f32,
    release: f32,
) -> PyResult<Vec<f32>> {
    Ok(py
        .allow_threads(|| mouth_openness_impl(&samples, sample_rate, fps, frames, attack, release)))
}

/// How many threads rayon will use, so `--doctor` can report it.
#[pyfunction]
fn threads() -> usize {
    rayon::current_num_threads()
}

#[pymodule]
fn video_gen_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(resize_crop, m)?)?;
    m.add_function(wrap_pyfunction!(blend_box, m)?)?;
    m.add_function(wrap_pyfunction!(mouth_openness, m)?)?;
    m.add_function(wrap_pyfunction!(threads, m)?)?;
    Ok(())
}
