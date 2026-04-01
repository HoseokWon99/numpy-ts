/**
 * WASM-accelerated element-wise reciprocal.
 *
 * Unary: out[i] = 1.0 / a[i]
 * Returns null if WASM can't handle this case.
 */

import {
  reciprocal_f64,
  reciprocal_f32,
  reciprocal_i64_f64,
  reciprocal_i32_f64,
  reciprocal_i16_f64,
  reciprocal_i8_f64,
} from './bins/reciprocal.wasm';
import { wasmMalloc, resetScratchAllocator, resolveInputPtr } from './runtime';
import { ArrayStorage } from '../storage';
import type { DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type UnaryFn = (aPtr: number, outPtr: number, N: number) => void;

// Float kernels (same-dtype in/out)
const floatKernels: Partial<Record<DType, UnaryFn>> = {
  float64: reciprocal_f64,
  float32: reciprocal_f32,
  // float16 excluded: f16→f32 conversion overhead makes JS path faster
};

// Integer-to-f64 kernels (int in, f64 out)
const intKernels: Partial<Record<DType, UnaryFn>> = {
  int64: reciprocal_i64_f64,
  uint64: reciprocal_i64_f64,
  int32: reciprocal_i32_f64,
  uint32: reciprocal_i32_f64,
  int16: reciprocal_i16_f64,
  uint16: reciprocal_i16_f64,
  int8: reciprocal_i8_f64,
  uint8: reciprocal_i8_f64,
};

type AnyTypedArrayCtor = new (length: number) => TypedArray;
const ctorMap: Partial<Record<DType, AnyTypedArrayCtor>> = {
  float64: Float64Array,
  float32: Float32Array,
  int64: BigInt64Array,
  uint64: BigUint64Array,
  int32: Int32Array,
  uint32: Uint32Array,
  int16: Int16Array,
  uint16: Uint16Array,
  int8: Int8Array,
  uint8: Uint8Array,
};

export function wasmReciprocal(a: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous) return null;
  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;

  // Try float kernel (same-dtype output)
  const floatKernel = floatKernels[dtype];
  if (floatKernel) {
    const Ctor = ctorMap[dtype]!;
    const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
    const outBytes = size * bpe;

    const outRegion = wasmMalloc(outBytes);
    if (!outRegion) return null;

    wasmConfig.wasmCallCount++;

    resetScratchAllocator();
    const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);

    floatKernel(aPtr, outRegion.ptr, size);

    return ArrayStorage.fromWasmRegion(
      Array.from(a.shape),
      dtype,
      outRegion,
      size,
      Ctor as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
    );
  }

  // Try integer-to-f64 kernel
  const intKernel = intKernels[dtype];
  const InCtor = ctorMap[dtype];
  if (!intKernel || !InCtor) return null;

  const inBpe = (InCtor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const outBytes = size * 8;

  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;

  resetScratchAllocator();
  const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, inBpe);

  intKernel(aPtr, outRegion.ptr, size);

  return ArrayStorage.fromWasmRegion(
    Array.from(a.shape),
    'float64',
    outRegion,
    size,
    Float64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );
}
