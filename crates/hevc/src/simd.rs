//! Runtime choice of the instruction set the hot loops run with.
//!
//! The pixel kernels (inverse transforms, interpolation, weighted prediction, in-loop filters)
//! are written as plain loops over fixed-size blocks that the compiler vectorises. Built for the
//! baseline target (x86-64 = SSE2) they only get 128-bit vectors and no 32-bit vector multiply;
//! [`fearless_simd`] compiles each kernel again for SSE4.2, AVX2 and AVX-512 (NEON on ARM, SIMD128
//! on WASM) and [`run`] calls the best one this CPU has, with no `unsafe` here. Integer arithmetic
//! gives the same results whatever the vector width, so the output stays bit-exact.

use fearless_simd::Level;
use std::sync::OnceLock;

/// The best SIMD level of this CPU (detected once).
#[inline(always)]
pub fn level() -> Level {
    static LEVEL: OnceLock<Level> = OnceLock::new();
    *LEVEL.get_or_init(Level::new)
}

/// Run `f` compiled for the best SIMD level of this CPU. Everything `f` calls that should use those
/// instructions must be `#[inline(always)]` (it is then compiled into this level's copy).
#[inline(always)]
pub fn run<R>(f: impl FnOnce() -> R) -> R {
    fearless_simd::dispatch!(level(), simd => fearless_simd::Simd::vectorize(simd, f))
}
