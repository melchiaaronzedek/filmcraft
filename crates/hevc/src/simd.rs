//! The place the hot loops run through, so the instruction set they are compiled for can be
//! chosen at run time.
//!
//! The pixel kernels (inverse transforms, interpolation, weighted prediction, in-loop filters)
//! are written as plain loops over fixed-size blocks that the compiler vectorises for the build
//! target. [`run`] calls them as they are; it is the single place a runtime choice of SSE4.2 /
//! AVX2 / AVX-512 (NEON, SIMD128) copies of those loops would go. Everything a kernel calls is
//! `#[inline(always)]` so such a copy would contain it.

/// Run `f` (see the module documentation).
#[inline(always)]
pub fn run<R>(f: impl FnOnce() -> R) -> R {
    f()
}
