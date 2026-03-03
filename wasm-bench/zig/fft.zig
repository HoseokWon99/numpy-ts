// FFT WASM kernels: rfft2, irfft2
// Uses Cooley-Tukey radix-2 DIT for power-of-2 sizes, Bluestein's for arbitrary sizes.
// Complex numbers stored as interleaved [re, im, re, im, ...].

const std = @import("std");
const math = std.math;

// ─── Helpers ────────────────────────────────────────────────────────────────

fn nextPow2(n: usize) usize {
    var v: usize = 1;
    while (v < n) v <<= 1;
    return v;
}

fn isPow2(n: usize) bool {
    return n > 0 and (n & (n - 1)) == 0;
}

// ─── In-place radix-2 Cooley-Tukey FFT ─────────────────────────────────────
// data = interleaved complex [re0, im0, re1, im1, ...], length = 2*N
// N must be a power of 2. inverse: false=forward, true=inverse

fn fftPow2(data: [*]f64, N: usize, inverse: bool) void {
    // Bit-reversal permutation
    var j: usize = 0;
    for (1..N) |i| {
        var bit = N >> 1;
        while (j & bit != 0) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if (i < j) {
            // Swap complex elements i and j
            const ti = i * 2;
            const tj = j * 2;
            const tr = data[ti];
            const timg = data[ti + 1];
            data[ti] = data[tj];
            data[ti + 1] = data[tj + 1];
            data[tj] = tr;
            data[tj + 1] = timg;
        }
    }

    // Butterfly passes
    const sign: f64 = if (inverse) 1.0 else -1.0;
    var len: usize = 2;
    while (len <= N) : (len <<= 1) {
        const half = len >> 1;
        const angle = sign * 2.0 * math.pi / @as(f64, @floatFromInt(len));
        const wr_step = @cos(angle);
        const wi_step = @sin(angle);

        var start: usize = 0;
        while (start < N) : (start += len) {
            var wr: f64 = 1.0;
            var wi: f64 = 0.0;
            for (0..half) |k| {
                const u_idx = (start + k) * 2;
                const v_idx = (start + k + half) * 2;
                const ur = data[u_idx];
                const ui = data[u_idx + 1];
                const vr = data[v_idx];
                const vi = data[v_idx + 1];

                const tvr = wr * vr - wi * vi;
                const tvi = wr * vi + wi * vr;

                data[u_idx] = ur + tvr;
                data[u_idx + 1] = ui + tvi;
                data[v_idx] = ur - tvr;
                data[v_idx + 1] = ui - tvi;

                const new_wr = wr * wr_step - wi * wi_step;
                wi = wr * wi_step + wi * wr_step;
                wr = new_wr;
            }
        }
    }

    // Scale for inverse
    if (inverse) {
        const scale = 1.0 / @as(f64, @floatFromInt(N));
        for (0..N * 2) |i| {
            data[i] *= scale;
        }
    }
}

// ─── Bluestein's FFT (arbitrary size N) ────────────────────────────────────
// scratch layout: chirp[2*P] + a_pad[2*P] + b_pad[2*P] where P = nextPow2(2*N-1)

fn bluesteinFft(input: [*]const f64, output: [*]f64, N: usize, inverse: bool, scratch: [*]f64) void {
    if (N <= 1) {
        if (N == 1) {
            output[0] = input[0];
            output[1] = input[1];
        }
        return;
    }

    if (isPow2(N)) {
        // Copy input to output and do in-place
        for (0..N * 2) |i| output[i] = input[i];
        fftPow2(output, N, inverse);
        return;
    }

    const P = nextPow2(2 * N - 1);
    const chirp = scratch;
    const a_pad = scratch + 2 * P;
    const b_pad = a_pad + 2 * P;

    const sign: f64 = if (inverse) -1.0 else 1.0;

    // Build chirp: chirp[k] = exp(sign * i * pi * k^2 / N)
    for (0..N) |k| {
        const angle = sign * math.pi * @as(f64, @floatFromInt(k * k)) / @as(f64, @floatFromInt(N));
        chirp[2 * k] = @cos(angle);
        chirp[2 * k + 1] = @sin(angle);
    }

    // a[k] = input[k] * conj(chirp[k])
    for (0..P * 2) |i| a_pad[i] = 0;
    for (0..N) |k| {
        const ir = input[2 * k];
        const ii = input[2 * k + 1];
        const cr = chirp[2 * k];
        const ci = -chirp[2 * k + 1]; // conjugate
        a_pad[2 * k] = ir * cr - ii * ci;
        a_pad[2 * k + 1] = ir * ci + ii * cr;
    }

    // b[0] = chirp[0], b[k] = b[P-k] = chirp[k] for k=1..N-1
    for (0..P * 2) |i| b_pad[i] = 0;
    b_pad[0] = chirp[0];
    b_pad[1] = chirp[1];
    for (1..N) |k| {
        b_pad[2 * k] = chirp[2 * k];
        b_pad[2 * k + 1] = chirp[2 * k + 1];
        b_pad[2 * (P - k)] = chirp[2 * k];
        b_pad[2 * (P - k) + 1] = chirp[2 * k + 1];
    }

    // Forward FFT both
    fftPow2(a_pad, P, false);
    fftPow2(b_pad, P, false);

    // Pointwise complex multiply
    for (0..P) |k| {
        const ar = a_pad[2 * k];
        const ai = a_pad[2 * k + 1];
        const br = b_pad[2 * k];
        const bi = b_pad[2 * k + 1];
        a_pad[2 * k] = ar * br - ai * bi;
        a_pad[2 * k + 1] = ar * bi + ai * br;
    }

    // Inverse FFT
    fftPow2(a_pad, P, true);

    // output[k] = a_pad[k] * conj(chirp[k])
    for (0..N) |k| {
        const ar = a_pad[2 * k];
        const ai = a_pad[2 * k + 1];
        const cr = chirp[2 * k];
        const ci = -chirp[2 * k + 1];
        output[2 * k] = ar * cr - ai * ci;
        output[2 * k + 1] = ar * ci + ai * cr;
    }

    // Scale for inverse
    if (inverse) {
        const scale = 1.0 / @as(f64, @floatFromInt(N));
        for (0..N * 2) |i| {
            output[i] *= scale;
        }
    }
}

// Scratch needed for one bluestein call: 6 * nextPow2(2*N-1) floats
fn bluesteinScratch(N: usize) usize {
    if (N <= 1) return 0;
    return 6 * nextPow2(2 * N - 1);
}

// ─── rfft2: M×N real → M×(N/2+1) complex ──────────────────────────────────

export fn rfft2_f64(inp: [*]const f64, out: [*]f64, scratch: [*]f64, m: u32, n: u32) void {
    const M = @as(usize, m);
    const N = @as(usize, n);
    const half_n = N / 2 + 1;

    // scratch layout: row_buf[2*N] + col_buf[2*M] + bluestein_scratch
    const row_buf = scratch;
    const col_buf = row_buf + 2 * N;
    const fft_scratch = col_buf + 2 * M;

    // Step 1: FFT each row, keep first half_n bins
    for (0..M) |row| {
        // Pack real → complex
        for (0..N) |j| {
            row_buf[2 * j] = inp[row * N + j];
            row_buf[2 * j + 1] = 0;
        }

        // FFT row in-place using col_buf as output, fft_scratch for bluestein
        bluesteinFft(row_buf, col_buf, N, false, fft_scratch);

        // Store first half_n bins into out[row, :]
        for (0..half_n) |j| {
            out[(row * half_n + j) * 2] = col_buf[2 * j];
            out[(row * half_n + j) * 2 + 1] = col_buf[2 * j + 1];
        }
    }

    // Step 2: FFT each column of the half-spectrum (length M)
    for (0..half_n) |col| {
        // Extract column
        for (0..M) |row| {
            col_buf[2 * row] = out[(row * half_n + col) * 2];
            col_buf[2 * row + 1] = out[(row * half_n + col) * 2 + 1];
        }

        // FFT column
        bluesteinFft(col_buf, row_buf, M, false, fft_scratch);

        // Write back
        for (0..M) |row| {
            out[(row * half_n + col) * 2] = row_buf[2 * row];
            out[(row * half_n + col) * 2 + 1] = row_buf[2 * row + 1];
        }
    }
}

// ─── irfft2: M×(N/2+1) complex → M×N real ─────────────────────────────────

export fn irfft2_f64(inp: [*]const f64, out: [*]f64, scratch: [*]f64, m: u32, n: u32) void {
    const M = @as(usize, m);
    const N = @as(usize, n);
    const half_n = N / 2 + 1;

    // scratch layout: work_copy[M*half_n*2] + full_row[2*N] + col_buf[2*M] + fft_scratch
    const work = scratch;
    const full_row = work + M * half_n * 2;
    const col_buf = full_row + 2 * N;
    const fft_scratch = col_buf + 2 * M;

    // Copy input to work (we'll modify it)
    for (0..M * half_n * 2) |i| work[i] = inp[i];

    // Step 1: IFFT each column (length M)
    for (0..half_n) |col| {
        // Extract column
        for (0..M) |row| {
            col_buf[2 * row] = work[(row * half_n + col) * 2];
            col_buf[2 * row + 1] = work[(row * half_n + col) * 2 + 1];
        }

        // IFFT column
        bluesteinFft(col_buf, full_row, M, true, fft_scratch);

        // Write back
        for (0..M) |row| {
            work[(row * half_n + col) * 2] = full_row[2 * row];
            work[(row * half_n + col) * 2 + 1] = full_row[2 * row + 1];
        }
    }

    // Step 2: IFFT each row, reconstruct full spectrum via Hermitian symmetry
    for (0..M) |row| {
        // First half_n bins from work
        for (0..half_n) |j| {
            full_row[2 * j] = work[(row * half_n + j) * 2];
            full_row[2 * j + 1] = work[(row * half_n + j) * 2 + 1];
        }

        // Hermitian symmetry: X[N-k] = conj(X[k])
        for (half_n..N) |j| {
            const mirror = N - j;
            full_row[2 * j] = full_row[2 * mirror];
            full_row[2 * j + 1] = -full_row[2 * mirror + 1];
        }

        // IFFT row
        bluesteinFft(full_row, col_buf, N, true, fft_scratch);

        // Extract real parts
        for (0..N) |j| {
            out[row * N + j] = col_buf[2 * j];
        }
    }
}
