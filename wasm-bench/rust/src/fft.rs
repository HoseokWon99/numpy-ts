// FFT WASM kernels: rfft2, irfft2
// Cooley-Tukey radix-2 DIT for power-of-2, Bluestein's for arbitrary sizes.
// Complex stored as interleaved [re, im, re, im, ...].

use libm::{cos, sin};

fn next_pow2(n: usize) -> usize {
    let mut v = 1;
    while v < n { v <<= 1; }
    v
}

fn is_pow2(n: usize) -> bool {
    n > 0 && (n & (n - 1)) == 0
}

const PI: f64 = core::f64::consts::PI;

// ─── In-place radix-2 Cooley-Tukey FFT ─────────────────────────────────────

unsafe fn fft_pow2(data: *mut f64, n: usize, inverse: bool) {
    // Bit-reversal permutation
    let mut j: usize = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            let ti = i * 2;
            let tj = j * 2;
            let tr = *data.add(ti);
            let timg = *data.add(ti + 1);
            *data.add(ti) = *data.add(tj);
            *data.add(ti + 1) = *data.add(tj + 1);
            *data.add(tj) = tr;
            *data.add(tj + 1) = timg;
        }
    }

    let sign: f64 = if inverse { 1.0 } else { -1.0 };
    let mut len: usize = 2;
    while len <= n {
        let half = len >> 1;
        let angle = sign * 2.0 * PI / (len as f64);
        let wr_step = cos(angle);
        let wi_step = sin(angle);

        let mut start: usize = 0;
        while start < n {
            let mut wr = 1.0f64;
            let mut wi = 0.0f64;
            for k in 0..half {
                let u_idx = (start + k) * 2;
                let v_idx = (start + k + half) * 2;
                let ur = *data.add(u_idx);
                let ui = *data.add(u_idx + 1);
                let vr = *data.add(v_idx);
                let vi = *data.add(v_idx + 1);

                let tvr = wr * vr - wi * vi;
                let tvi = wr * vi + wi * vr;

                *data.add(u_idx) = ur + tvr;
                *data.add(u_idx + 1) = ui + tvi;
                *data.add(v_idx) = ur - tvr;
                *data.add(v_idx + 1) = ui - tvi;

                let new_wr = wr * wr_step - wi * wi_step;
                wi = wr * wi_step + wi * wr_step;
                wr = new_wr;
            }
            start += len;
        }
        len <<= 1;
    }

    if inverse {
        let scale = 1.0 / (n as f64);
        for i in 0..n * 2 {
            *data.add(i) *= scale;
        }
    }
}

// ─── Bluestein's FFT ───────────────────────────────────────────────────────

unsafe fn bluestein_fft(input: *const f64, output: *mut f64, n: usize, inverse: bool, scratch: *mut f64) {
    if n <= 1 {
        if n == 1 {
            *output = *input;
            *output.add(1) = *input.add(1);
        }
        return;
    }

    if is_pow2(n) {
        for i in 0..n * 2 { *output.add(i) = *input.add(i); }
        fft_pow2(output, n, inverse);
        return;
    }

    let p = next_pow2(2 * n - 1);
    let chirp = scratch;
    let a_pad = scratch.add(2 * p);
    let b_pad = a_pad.add(2 * p);

    let sign: f64 = if inverse { -1.0 } else { 1.0 };

    // Build chirp
    for k in 0..n {
        let angle = sign * PI * ((k * k) as f64) / (n as f64);
        *chirp.add(2 * k) = cos(angle);
        *chirp.add(2 * k + 1) = sin(angle);
    }

    // a[k] = input[k] * conj(chirp[k])
    for i in 0..p * 2 { *a_pad.add(i) = 0.0; }
    for k in 0..n {
        let ir = *input.add(2 * k);
        let ii = *input.add(2 * k + 1);
        let cr = *chirp.add(2 * k);
        let ci = -*chirp.add(2 * k + 1);
        *a_pad.add(2 * k) = ir * cr - ii * ci;
        *a_pad.add(2 * k + 1) = ir * ci + ii * cr;
    }

    // b[0] = chirp[0], b[k] = b[P-k] = chirp[k]
    for i in 0..p * 2 { *b_pad.add(i) = 0.0; }
    *b_pad = *chirp;
    *b_pad.add(1) = *chirp.add(1);
    for k in 1..n {
        *b_pad.add(2 * k) = *chirp.add(2 * k);
        *b_pad.add(2 * k + 1) = *chirp.add(2 * k + 1);
        *b_pad.add(2 * (p - k)) = *chirp.add(2 * k);
        *b_pad.add(2 * (p - k) + 1) = *chirp.add(2 * k + 1);
    }

    fft_pow2(a_pad, p, false);
    fft_pow2(b_pad, p, false);

    // Pointwise multiply
    for k in 0..p {
        let ar = *a_pad.add(2 * k);
        let ai = *a_pad.add(2 * k + 1);
        let br = *b_pad.add(2 * k);
        let bi = *b_pad.add(2 * k + 1);
        *a_pad.add(2 * k) = ar * br - ai * bi;
        *a_pad.add(2 * k + 1) = ar * bi + ai * br;
    }

    fft_pow2(a_pad, p, true);

    // output[k] = a_pad[k] * conj(chirp[k])
    for k in 0..n {
        let ar = *a_pad.add(2 * k);
        let ai = *a_pad.add(2 * k + 1);
        let cr = *chirp.add(2 * k);
        let ci = -*chirp.add(2 * k + 1);
        *output.add(2 * k) = ar * cr - ai * ci;
        *output.add(2 * k + 1) = ar * ci + ai * cr;
    }

    if inverse {
        let scale = 1.0 / (n as f64);
        for i in 0..n * 2 {
            *output.add(i) *= scale;
        }
    }
}

// ─── rfft2: M×N real → M×(N/2+1) complex ──────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn rfft2_f64(inp: *const f64, out: *mut f64, scratch: *mut f64, m: u32, n: u32) {
    let rows = m as usize;
    let cols = n as usize;
    let half_n = cols / 2 + 1;

    let row_buf = scratch;
    let col_buf = row_buf.add(2 * cols);
    let fft_scratch = col_buf.add(2 * rows);

    // Step 1: FFT each row
    for row in 0..rows {
        for j in 0..cols {
            *row_buf.add(2 * j) = *inp.add(row * cols + j);
            *row_buf.add(2 * j + 1) = 0.0;
        }
        bluestein_fft(row_buf as *const f64, col_buf, cols, false, fft_scratch);
        for j in 0..half_n {
            *out.add((row * half_n + j) * 2) = *col_buf.add(2 * j);
            *out.add((row * half_n + j) * 2 + 1) = *col_buf.add(2 * j + 1);
        }
    }

    // Step 2: FFT each column
    for col in 0..half_n {
        for row in 0..rows {
            *col_buf.add(2 * row) = *out.add((row * half_n + col) * 2);
            *col_buf.add(2 * row + 1) = *out.add((row * half_n + col) * 2 + 1);
        }
        bluestein_fft(col_buf as *const f64, row_buf, rows, false, fft_scratch);
        for row in 0..rows {
            *out.add((row * half_n + col) * 2) = *row_buf.add(2 * row);
            *out.add((row * half_n + col) * 2 + 1) = *row_buf.add(2 * row + 1);
        }
    }
}

// ─── irfft2: M×(N/2+1) complex → M×N real ─────────────────────────────────

#[no_mangle]
pub unsafe extern "C" fn irfft2_f64(inp: *const f64, out: *mut f64, scratch: *mut f64, m: u32, n: u32) {
    let rows = m as usize;
    let cols = n as usize;
    let half_n = cols / 2 + 1;

    let work = scratch;
    let full_row = work.add(rows * half_n * 2);
    let col_buf = full_row.add(2 * cols);
    let fft_scratch = col_buf.add(2 * rows);

    // Copy input to work
    for i in 0..rows * half_n * 2 { *work.add(i) = *inp.add(i); }

    // Step 1: IFFT each column
    for col in 0..half_n {
        for row in 0..rows {
            *col_buf.add(2 * row) = *work.add((row * half_n + col) * 2);
            *col_buf.add(2 * row + 1) = *work.add((row * half_n + col) * 2 + 1);
        }
        bluestein_fft(col_buf as *const f64, full_row, rows, true, fft_scratch);
        for row in 0..rows {
            *work.add((row * half_n + col) * 2) = *full_row.add(2 * row);
            *work.add((row * half_n + col) * 2 + 1) = *full_row.add(2 * row + 1);
        }
    }

    // Step 2: IFFT each row with Hermitian reconstruction
    for row in 0..rows {
        for j in 0..half_n {
            *full_row.add(2 * j) = *work.add((row * half_n + j) * 2);
            *full_row.add(2 * j + 1) = *work.add((row * half_n + j) * 2 + 1);
        }
        for j in half_n..cols {
            let mirror = cols - j;
            *full_row.add(2 * j) = *full_row.add(2 * mirror);
            *full_row.add(2 * j + 1) = -*full_row.add(2 * mirror + 1);
        }
        bluestein_fft(full_row as *const f64, col_buf, cols, true, fft_scratch);
        for j in 0..cols {
            *out.add(row * cols + j) = *col_buf.add(2 * j);
        }
    }
}
