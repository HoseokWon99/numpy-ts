/**
 * WASM-accelerated FFT kernels.
 *
 * Provides 1D complex-to-complex FFT/IFFT and real FFT/IRFFT.
 * Uses Cooley-Tukey radix-2 for power-of-2 sizes, Bluestein's for arbitrary sizes.
 * Returns null if WASM can't handle this case.
 */

import {
  fft_c128,
  ifft_c128,
  fft_c64,
  ifft_c64,
  rfft_f64,
  irfft_f64,
  fft_scratch_size,
} from './bins/fft.wasm';
import { ensureMemory, resetAllocator, copyIn, alloc, copyOut } from './runtime';
import { ArrayStorage } from '../storage';
import type { TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 32;

/**
 * WASM-accelerated 1D forward complex FFT.
 * Input: contiguous complex128 (interleaved f64) or complex64 (interleaved f32).
 * Returns null if WASM can't handle.
 */
export function wasmFft(a: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const size = a.size; // number of complex elements
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  if (dtype === 'complex128') {
    const scratchN = fft_scratch_size(size);
    const dataLen = size * 2; // interleaved re,im
    const inBytes = dataLen * 8;
    const outBytes = dataLen * 8;
    const scratchBytes = scratchN * 8;

    ensureMemory(inBytes + outBytes + scratchBytes);
    resetAllocator();

    const aData = a.data.subarray(a.offset * 2, (a.offset + size) * 2) as TypedArray;
    const inPtr = copyIn(aData);
    const outPtr = alloc(outBytes);
    const scratchPtr = alloc(scratchBytes);

    fft_c128(inPtr, outPtr, scratchPtr, size);

    const outData = copyOut(
      outPtr,
      dataLen,
      Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
    );
    return ArrayStorage.fromData(outData, Array.from(a.shape), 'complex128');
  }

  if (dtype === 'complex64') {
    const scratchN = 4 * size + fft_scratch_size(size);
    const dataLen = size * 2;
    const inBytes = dataLen * 4; // f32
    const outBytes = dataLen * 4;
    const scratchBytes = scratchN * 8; // scratch is always f64

    ensureMemory(inBytes + outBytes + scratchBytes);
    resetAllocator();

    const aData = a.data.subarray(a.offset * 2, (a.offset + size) * 2) as TypedArray;
    const inPtr = copyIn(aData);
    const outPtr = alloc(outBytes);
    const scratchPtr = alloc(scratchBytes);

    fft_c64(inPtr, outPtr, scratchPtr, size);

    const outData = copyOut(
      outPtr,
      dataLen,
      Float32Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
    );
    return ArrayStorage.fromData(outData, Array.from(a.shape), 'complex64');
  }

  return null;
}

/**
 * WASM-accelerated 1D inverse complex FFT.
 */
export function wasmIfft(a: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  if (dtype === 'complex128') {
    const scratchN = fft_scratch_size(size);
    const dataLen = size * 2;
    const inBytes = dataLen * 8;
    const outBytes = dataLen * 8;
    const scratchBytes = scratchN * 8;

    ensureMemory(inBytes + outBytes + scratchBytes);
    resetAllocator();

    const aData = a.data.subarray(a.offset * 2, (a.offset + size) * 2) as TypedArray;
    const inPtr = copyIn(aData);
    const outPtr = alloc(outBytes);
    const scratchPtr = alloc(scratchBytes);

    ifft_c128(inPtr, outPtr, scratchPtr, size);

    const outData = copyOut(
      outPtr,
      dataLen,
      Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
    );
    return ArrayStorage.fromData(outData, Array.from(a.shape), 'complex128');
  }

  if (dtype === 'complex64') {
    const scratchN = 4 * size + fft_scratch_size(size);
    const dataLen = size * 2;
    const inBytes = dataLen * 4;
    const outBytes = dataLen * 4;
    const scratchBytes = scratchN * 8;

    ensureMemory(inBytes + outBytes + scratchBytes);
    resetAllocator();

    const aData = a.data.subarray(a.offset * 2, (a.offset + size) * 2) as TypedArray;
    const inPtr = copyIn(aData);
    const outPtr = alloc(outBytes);
    const scratchPtr = alloc(scratchBytes);

    ifft_c64(inPtr, outPtr, scratchPtr, size);

    const outData = copyOut(
      outPtr,
      dataLen,
      Float32Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
    );
    return ArrayStorage.fromData(outData, Array.from(a.shape), 'complex64');
  }

  return null;
}

/**
 * WASM-accelerated 1D real-to-complex FFT.
 * Input: contiguous float64 real array. Output: complex128 with n/2+1 elements.
 */
export function wasmRfft(a: ArrayStorage, n: number): ArrayStorage | null {
  if (!a.isCContiguous) return null;
  if (a.dtype !== 'float64') return null;
  if (n < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const halfN = Math.floor(n / 2) + 1;
  // scratch: 2*n (pack to complex) + 2*n (bluestein output) + fft_scratch_size(n)
  const scratchN = 4 * n + fft_scratch_size(n);
  const inBytes = n * 8;
  const outBytes = halfN * 2 * 8;
  const scratchBytes = scratchN * 8;

  ensureMemory(inBytes + outBytes + scratchBytes);
  resetAllocator();

  const aData = a.data.subarray(a.offset, a.offset + n) as TypedArray;
  const inPtr = copyIn(aData);
  const outPtr = alloc(outBytes);
  const scratchPtr = alloc(scratchBytes);

  rfft_f64(inPtr, outPtr, scratchPtr, n);

  const outData = copyOut(
    outPtr,
    halfN * 2,
    Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );

  const outShape = Array.from(a.shape);
  outShape[outShape.length - 1] = halfN;
  return ArrayStorage.fromData(outData, outShape, 'complex128');
}

/**
 * WASM-accelerated 1D complex-to-real inverse FFT.
 * Input: contiguous complex128 with n_half elements. Output: float64 with n_out elements.
 */
export function wasmIrfft(a: ArrayStorage, nOut: number): ArrayStorage | null {
  if (!a.isCContiguous) return null;
  if (a.dtype !== 'complex128') return null;

  const nHalf = a.size;
  if (nOut < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  // scratch: 2*nOut (full spectrum) + 2*nOut (bluestein output) + fft_scratch_size(nOut)
  const scratchN = 4 * nOut + fft_scratch_size(nOut);
  const inBytes = nHalf * 2 * 8;
  const outBytes = nOut * 8;
  const scratchBytes = scratchN * 8;

  ensureMemory(inBytes + outBytes + scratchBytes);
  resetAllocator();

  const aData = a.data.subarray(a.offset * 2, (a.offset + nHalf) * 2) as TypedArray;
  const inPtr = copyIn(aData);
  const outPtr = alloc(outBytes);
  const scratchPtr = alloc(scratchBytes);

  irfft_f64(inPtr, outPtr, scratchPtr, nHalf, nOut);

  const outData = copyOut(
    outPtr,
    nOut,
    Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );

  const outShape = Array.from(a.shape);
  outShape[outShape.length - 1] = nOut;
  return ArrayStorage.fromData(outData, outShape, 'float64');
}
