//! Scaling (8.6.3), transform skip and inverse transforms (8.6.4): DCT 4..32 and DST 4x4.

use crate::tables::{DST_MATRIX, TRANS_MATRIX};

/// Inverse transform of an n x n block of scaled coefficients `c` (row-major, `c[y * n + x]`) in place,
/// producing residuals after the final `bd_shift` rounding (8-299). `max_x` / `max_y` bound the
/// non-zero coefficients (inclusive). `dst`: DST-VII (intra 4x4 luma).
///
/// The DCT is evaluated as a partial butterfly: the basis functions of the N-point DCT (rows
/// `j * 32 / N` of the 32x32 matrix of 8.6.4.2, first N columns) are symmetric for even `j` and
/// antisymmetric for odd `j` about the middle, so with `E[k]` the sum over even coefficients and
/// `O[k]` the sum over odd ones (k < N/2), `out[k] = E[k] + O[k]` and `out[N-1-k] = E[k] - O[k]`;
/// the even coefficients form the N/2-point transform (rows of the same matrix), recursively.
/// That is a third of the multiplications of the direct product at 16 points and a quarter at 32.
/// Zero coefficients (most of them) are skipped, and the sizes are compile-time constants so the
/// inner loops over contiguous basis rows vectorise. The integer sums are exact (|sum| < 2^27), so
/// the result equals the direct matrix product of 8.6.4.2 in any summation order.
pub fn inverse_transform(c: &mut [i32], n: usize, dst: bool, max_x: usize, max_y: usize, bd_shift: u32) {
    // compiled per instruction set, the best one at run time (`crate::simd`)
    match n {
        4 if dst => inverse_dst4(c, max_x, max_y, bd_shift),
        4 => crate::simd::run(
            #[inline(always)]
            || inverse::<4>(c, max_x, max_y, bd_shift),
        ),
        8 => crate::simd::run(
            #[inline(always)]
            || inverse::<8>(c, max_x, max_y, bd_shift),
        ),
        16 => crate::simd::run(
            #[inline(always)]
            || inverse::<16>(c, max_x, max_y, bd_shift),
        ),
        _ => crate::simd::run(
            #[inline(always)]
            || inverse::<32>(c, max_x, max_y, bd_shift),
        ),
    }
}

/// Basis row `j` of the N-point DCT: row `j * 32 / N` of the 32x32 matrix.
#[inline(always)]
fn basis<const N: usize>(j: usize) -> &'static [i8] {
    &TRANS_MATRIX[j * (32 / N)][..N / 2]
}

/// `acc[k] += m[k] * d` over the first half of a basis row.
#[inline(always)]
fn mac<const H: usize>(acc: &mut [i32; H], m: &[i8], d: i32) {
    if let Some(m) = m.first_chunk::<H>() {
        for k in 0..H {
            acc[k] += m[k] as i32 * d;
        }
    }
}

/// One N-point inverse DCT (N = `c.len()`: 4, 8, 16 or 32) of the coefficients `c` (indices
/// `0..=last` may be non-zero) into `out` (same length).
#[inline(always)]
fn dct_1d(c: &[i32], last: usize, out: &mut [i32]) {
    match c.len() {
        4 => butterfly::<4, 2>(c, last, out, dct_2),
        8 => butterfly::<8, 4>(c, last, out, dct_4),
        16 => butterfly::<16, 8>(c, last, out, dct_8),
        _ => butterfly::<32, 16>(c, last, out, dct_16),
    }
}

/// The 2-point even part of the 4-point transform: rows 0 and 16 of the 32x32 matrix
/// ([64, 64] and [64, -64]).
#[inline(always)]
fn dct_2(c: &[i32], _last: usize, e: &mut [i32]) {
    let (a, b) = (64 * c.first().copied().unwrap_or(0), 64 * c.get(1).copied().unwrap_or(0));
    if let [e0, e1, ..] = e {
        *e0 = a + b;
        *e1 = a - b;
    }
}

#[inline(always)]
fn dct_4(c: &[i32], last: usize, e: &mut [i32]) {
    butterfly::<4, 2>(c, last, e, dct_2)
}

#[inline(always)]
fn dct_8(c: &[i32], last: usize, e: &mut [i32]) {
    butterfly::<8, 4>(c, last, e, dct_4)
}

#[inline(always)]
fn dct_16(c: &[i32], last: usize, e: &mut [i32]) {
    butterfly::<16, 8>(c, last, e, dct_8)
}

/// The partial butterfly of an N = 2H point transform, with `even` the H-point transform of the
/// even coefficients.
#[inline(always)]
fn butterfly<const N: usize, const H: usize>(c: &[i32], last: usize, out: &mut [i32], even: fn(&[i32], usize, &mut [i32])) {
    let (Some(c), Some(out)) = (c.first_chunk::<N>(), out.first_chunk_mut::<N>()) else { return };
    // odd part: O[k] = sum over odd j of M_N[j][k] * c[j], k < H
    let mut o = [0i32; H];
    let mut j = 1;
    while j <= last.min(N - 1) {
        let d = c[j];
        if d != 0 {
            mac::<H>(&mut o, basis::<N>(j), d);
        }
        j += 2;
    }
    // even part: the H-point transform of c[0], c[2], ...
    let mut ce = [0i32; H];
    for (k, v) in ce.iter_mut().enumerate() {
        *v = c[2 * k];
    }
    let mut e = [0i32; H];
    even(&ce, last / 2, &mut e);
    for k in 0..H {
        out[k] = e[k] + o[k];
        out[N - 1 - k] = e[k] - o[k];
    }
}

#[inline(always)]
fn inverse<const N: usize>(c: &mut [i32], max_x: usize, max_y: usize, bd_shift: u32) {
    if c.len() < N * N {
        return;
    }
    let (max_x, max_y) = (max_x.min(N - 1), max_y.min(N - 1));
    // 1. columns (first stage of 8.6.4.2): e[x][y] = sum_j M[j][y] * d[x][j], clipped after
    //    (x + 64) >> 7; stored transposed (col[x][y]) for the second stage
    let mut col = [[0i32; N]; N];
    let mut input = [0i32; N];
    for x in 0..=max_x {
        for j in 0..=max_y {
            input[j] = c[j * N + x];
        }
        for v in input.iter_mut().skip(max_y + 1) {
            *v = 0;
        }
        let acc = &mut col[x];
        dct_1d(&input, max_y, acc);
        for a in acc.iter_mut() {
            *a = ((*a + 64) >> 7).clamp(-32768, 32767);
        }
    }
    // 2. rows: r[x][y] = sum_j M[j][x] * g[j][y], g[j][y] = col[j][y] (only j <= max_x non-zero)
    let round = 1 << (bd_shift - 1);
    let mut out = [0i32; N];
    for y in 0..N {
        for (j, v) in input.iter_mut().enumerate() {
            *v = if j <= max_x { col[j][y] } else { 0 };
        }
        dct_1d(&input, max_x, &mut out);
        for (r, o) in c[y * N..y * N + N].iter_mut().zip(out) {
            *r = (o + round) >> bd_shift;
        }
    }
}

/// The 4x4 DST-VII of intra luma (no symmetry to exploit: the direct product, 16 terms per sample).
fn inverse_dst4(c: &mut [i32], max_x: usize, max_y: usize, bd_shift: u32) {
    if c.len() < 16 {
        return;
    }
    let (max_x, max_y) = (max_x.min(3), max_y.min(3));
    let mut col = [[0i32; 4]; 4];
    for x in 0..=max_x {
        let acc = &mut col[x];
        for j in 0..=max_y {
            let d = c[j * 4 + x];
            if d != 0 {
                for (a, &m) in acc.iter_mut().zip(&DST_MATRIX[j]) {
                    *a += m as i32 * d;
                }
            }
        }
        for a in acc.iter_mut() {
            *a = ((*a + 64) >> 7).clamp(-32768, 32767);
        }
    }
    let round = 1 << (bd_shift - 1);
    for y in 0..4 {
        let mut out = [round; 4];
        for (j, g) in col.iter().enumerate().take(max_x + 1) {
            let g = g[y];
            if g != 0 {
                for (o, &m) in out.iter_mut().zip(&DST_MATRIX[j]) {
                    *o += m as i32 * g;
                }
            }
        }
        for (r, o) in c[y * 4..y * 4 + 4].iter_mut().zip(out) {
            *r = o >> bd_shift;
        }
    }
}

/// DC-only inverse DCT: every residual sample gets the same value.
pub fn inverse_dc(dc: i32, bd_shift: u32) -> i32 {
    let e = ((64 * dc + 64) >> 7).clamp(-32768, 32767);
    (64 * e + (1 << (bd_shift - 1))) >> bd_shift
}

/// Transform skip residual (8-298, 8-299): r = (d << tsShift + round) >> bdShift.
pub fn transform_skip(c: &mut [i32], n: usize, bd_shift: u32) {
    let ts_shift = 5 + n.trailing_zeros();
    let round = 1 << (bd_shift - 1);
    for v in c[..n * n].iter_mut() {
        *v = ((*v << ts_shift) + round) >> bd_shift;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The direct matrix product of 8.6.4.2 (the original implementation), as the reference.
    fn reference(c: &mut [i32], n: usize, dst: bool, max_x: usize, max_y: usize, bd_shift: u32) {
        let mut tmp = [0i32; 32 * 32];
        let step = 32 / n;
        for x in 0..=max_x {
            for y in 0..n {
                let mut s = 0i32;
                for j in 0..=max_y {
                    let m = if dst { DST_MATRIX[j][y] as i32 } else { TRANS_MATRIX[j * step][y] as i32 };
                    s += m * c[j * n + x];
                }
                tmp[y * n + x] = ((s + 64) >> 7).clamp(-32768, 32767);
            }
        }
        let round = 1 << (bd_shift - 1);
        for y in 0..n {
            for x in 0..n {
                let mut s = 0i32;
                for j in 0..=max_x {
                    let m = if dst { DST_MATRIX[j][x] as i32 } else { TRANS_MATRIX[j * step][x] as i32 };
                    s += m * tmp[y * n + j];
                }
                c[y * n + x] = (s + round) >> bd_shift;
            }
        }
    }

    #[test]
    fn matches_the_direct_matrix_product() {
        // sparse and dense blocks, extreme coefficients, every size, DST, 8- and 10-bit shifts
        let mut rng = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for &(n, dst) in &[(4usize, true), (4, false), (8, false), (16, false), (32, false)] {
            for case in 0..400 {
                let (max_x, max_y) = ((next() as usize) % n, (next() as usize) % n);
                let density = [2u64, 8, 50, 100][case % 4];
                let mut c = vec![0i32; n * n];
                for y in 0..=max_y {
                    for x in 0..=max_x {
                        if next() % 100 < density {
                            c[y * n + x] = match next() % 8 {
                                0 => 32767,
                                1 => -32768,
                                _ => (next() % 2001) as i32 - 1000,
                            };
                        }
                    }
                }
                for bd_shift in [12u32, 10] {
                    let (mut a, mut b) = (c.clone(), c.clone());
                    inverse_transform(&mut a, n, dst, max_x, max_y, bd_shift);
                    reference(&mut b, n, dst, max_x, max_y, bd_shift);
                    assert_eq!(a, b, "n {n} dst {dst} max ({max_x}, {max_y}) shift {bd_shift}");
                }
            }
        }
    }

    /// The symmetry the butterfly relies on, for every transform size: basis row j of the N-point
    /// DCT is symmetric about the middle for even j and antisymmetric for odd j.
    #[test]
    fn every_size_has_the_butterfly_symmetry() {
        for n in [4usize, 8, 16, 32] {
            for j in 0..n {
                let row = &TRANS_MATRIX[j * (32 / n)];
                for k in 0..n / 2 {
                    let (a, b) = (row[k] as i32, row[n - 1 - k] as i32);
                    assert_eq!(a, if j % 2 == 0 { b } else { -b }, "n {n} row {j} col {k}");
                }
            }
        }
    }

    #[test]
    fn dc_only_matches_full_transform() {
        for &n in &[4usize, 8, 16, 32] {
            for dc in [-300, -1, 1, 7, 64, 1000] {
                let mut c = vec![0i32; n * n];
                c[0] = dc;
                inverse_transform(&mut c, n, false, 0, 0, 12);
                let v = inverse_dc(dc, 12);
                assert!(c.iter().all(|&x| x == v), "n {n} dc {dc}");
            }
        }
    }

    #[test]
    fn bounded_region_matches_full() {
        let n = 16;
        let mut c = vec![0i32; n * n];
        c[0] = 50;
        c[1] = -20;
        c[2 * n + 3] = 9;
        let mut a = c.clone();
        inverse_transform(&mut a, n, false, 3, 2, 12);
        let mut b = c.clone();
        inverse_transform(&mut b, n, false, n - 1, n - 1, 12);
        assert_eq!(a, b);
    }
}
