/**
 * WASM-accelerated stride-2 extraction for complex real/imag parts.
 *
 * Complex data is interleaved: [re0, im0, re1, im1, ...]
 * These kernels extract the real or imaginary lane in one SIMD pass.
 */

import {
  extract_real_f64,
  extract_imag_f64,
  extract_real_f32,
  extract_imag_f32,
} from './bins/complex_extract.wasm';
import { wasmMalloc, resetScratchAllocator, resolveInputPtr } from './runtime';
import { ArrayStorage } from '../storage';
import { getComplexComponentDType, type TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

/**
 * Extract real parts from a complex array.
 * Returns null if WASM can't handle.
 */
export function wasmReal(a: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const dtype = a.dtype;
  const isC128 = dtype === 'complex128';
  const isC64 = dtype === 'complex64';
  if (!isC128 && !isC64) return null;

  const size = a.size; // number of complex elements
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const componentBpe = isC128 ? 8 : 4;
  const outBytes = size * componentBpe;
  const Ctor = isC128 ? Float64Array : Float32Array;
  const kernel = isC128 ? extract_real_f64 : extract_real_f32;
  const outDtype = getComplexComponentDType(dtype);

  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  // Source is interleaved [re, im, re, im, ...] — physical size is 2× logical
  const srcPtr = resolveInputPtr(
    a.data,
    a.isWasmBacked,
    a.wasmPtr,
    a.offset * 2, // offset in component elements (not complex elements)
    size * 2, // physical element count
    componentBpe
  );

  kernel(srcPtr, outRegion.ptr, size);

  return ArrayStorage.fromWasmRegion(
    Array.from(a.shape),
    outDtype,
    outRegion,
    size,
    Ctor as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );
}

/**
 * Extract imaginary parts from a complex array.
 * Returns null if WASM can't handle.
 */
export function wasmImag(a: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const dtype = a.dtype;
  const isC128 = dtype === 'complex128';
  const isC64 = dtype === 'complex64';
  if (!isC128 && !isC64) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const componentBpe = isC128 ? 8 : 4;
  const outBytes = size * componentBpe;
  const Ctor = isC128 ? Float64Array : Float32Array;
  const kernel = isC128 ? extract_imag_f64 : extract_imag_f32;
  const outDtype = getComplexComponentDType(dtype);

  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  const srcPtr = resolveInputPtr(
    a.data,
    a.isWasmBacked,
    a.wasmPtr,
    a.offset * 2,
    size * 2,
    componentBpe
  );

  kernel(srcPtr, outRegion.ptr, size);

  return ArrayStorage.fromWasmRegion(
    Array.from(a.shape),
    outDtype,
    outRegion,
    size,
    Ctor as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );
}
