//! Decode an Annex-B H.265 file to raw planar YUV (yuv420p / yuv420p10le / yuv422p /
//! yuv422p10le): `cargo run --release -p filmcraft-hevc --example hevcdec -- in.hevc out.yuv`
//!
//! The stream is fed access unit by access unit and every picture is written (and dropped) as it
//! comes out, so memory use is the decoder's own. Environment:
//! - `HEVC_THREADS=n`: worker threads (default: every core, up to 16)
//! - `HEVC_LOOPS=k`: decode the file k times (benchmarking)
//! - `HEVC_DRAFT=1`: draft mode (non-reference pictures skip deblocking and SAO)
//! - `HEVC_MEM_MB=m`: limit the pictures in flight to about m MB ([`Decoder::set_memory_limit`])

use filmcraft_hevc::{Decoder, Plane};
use std::io::Write;
use std::sync::Mutex;

/// Output planes handed back for reuse, as the app's frame pool does (`filmcraft_frame::pool`):
/// without it every picture's planes are fresh allocations whose page faults are counted as
/// decoding time.
static POOL_U8: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());
static POOL_U16: Mutex<Vec<Vec<u16>>> = Mutex::new(Vec::new());

fn take<T>(pool: &Mutex<Vec<Vec<T>>>, len: usize) -> Vec<T> {
    let mut p = pool.lock().unwrap();
    match p.iter().position(|b| b.capacity() >= len) {
        Some(i) => p.swap_remove(i),
        None => Vec::with_capacity(len),
    }
}

fn take_u8(len: usize) -> Vec<u8> {
    take(&POOL_U8, len)
}

fn take_u16(len: usize) -> Vec<u16> {
    take(&POOL_U16, len)
}

fn give(plane: Plane) {
    match plane {
        Plane::U8(v) => POOL_U8.lock().unwrap().push(v),
        Plane::U16(v) => POOL_U16.lock().unwrap().push(v),
    }
}

/// Access units of an Annex-B stream: a new one starts at a VPS / SPS / PPS / AUD / prefix SEI
/// after a VCL NAL unit, or at a VCL NAL unit with first_slice_segment_in_pic_flag.
fn access_units(data: &[u8]) -> Vec<&[u8]> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 3 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            starts.push((if i > 0 && data[i - 1] == 0 { i - 1 } else { i }, i + 3));
            i += 3;
        } else {
            i += 1;
        }
    }
    let (mut aus, mut au_start, mut seen_vcl) = (Vec::new(), 0usize, false);
    for &(sc, payload) in &starts {
        if payload + 2 >= data.len() {
            continue;
        }
        let t = (data[payload] >> 1) & 0x3f;
        let vcl = t < 32;
        if seen_vcl && (matches!(t, 32..=35 | 39) || (vcl && data[payload + 2] & 0x80 != 0)) {
            aus.push(&data[au_start..sc]);
            au_start = sc;
            seen_vcl = false;
        }
        seen_vcl |= vcl;
    }
    if au_start < data.len() {
        aus.push(&data[au_start..]);
    }
    aus
}

/// Peak resident memory of this process in MB (Linux; 0 elsewhere).
fn peak_rss_mb() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| s.lines().find(|l| l.starts_with("VmHWM:")).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<u64>().ok()))
        .map_or(0, |kb| kb / 1024)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: hevcdec <in.hevc> [out.yuv]");
        std::process::exit(2);
    }
    let data = std::fs::read(&args[1]).expect("read input");
    let env = |k: &str| std::env::var(k).ok().and_then(|s| s.parse::<usize>().ok());
    let threads = env("HEVC_THREADS").unwrap_or(0);
    let loops = env("HEVC_LOOPS").unwrap_or(1).max(1);
    let mem = env("HEVC_MEM_MB").map(|m| m << 20);
    let mut out = args.get(2).map(|p| std::io::BufWriter::with_capacity(1 << 24, std::fs::File::create(p).expect("create output")));
    let aus = access_units(&data);
    filmcraft_hevc::set_plane_allocators(take_u8, take_u16);
    let t0 = std::time::Instant::now();
    let mut count = 0usize;
    let mut in_flight = 0;
    for l in 0..loops {
        let mut dec = if threads > 0 { Decoder::with_threads(threads) } else { Decoder::new() };
        dec.set_draft(std::env::var_os("HEVC_DRAFT").is_some());
        dec.set_memory_limit(mem);
        let mut emit = |pics: Vec<filmcraft_hevc::Picture>| {
            for p in pics {
                count += 1;
                if let (Some(w), 0) = (out.as_mut(), l) {
                    for plane in [&p.y, &p.u, &p.v] {
                        w.write_all(&plane.to_le_bytes()).expect("write output");
                    }
                }
                give(p.y);
                give(p.u);
                give(p.v);
            }
        };
        for (i, au) in aus.iter().enumerate() {
            emit(dec.decode(au, i as i64).expect("decode"));
        }
        in_flight = dec.in_flight_limit();
        emit(dec.flush());
        if let Some(e) = dec.take_error() {
            eprintln!("decode error: {e}");
            std::process::exit(1);
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    if let Some(w) = out.as_mut() {
        w.flush().expect("write output");
    }
    eprintln!("{count} pictures in {secs:.3}s ({:.1} fps); {in_flight} pictures in flight; peak resident {} MB", count as f64 / secs, peak_rss_mb());
}
