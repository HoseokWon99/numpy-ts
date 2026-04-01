/**
 * WASM-accelerated L2 vector norm: sqrt(sum(x[i]^2)).
 * Returns null if WASM can't handle.
 */

import { vector_norm2_f64, vector_norm2_f32 } from './bins/vector_norm.wasm';
import { resetScratchAllocator, resolveInputPtr, scratchCopyIn, f16ToF32Input } from './runtime';
import { ArrayStorage } from '../storage';
import { isComplexDType, type DType, type TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type NormFn = (aPtr: number, N: number) => number;

const kernels: Partial<Record<DType, { fn: NormFn; bpe: number }>> = {
  float64: { fn: vector_norm2_f64, bpe: 8 },
  float32: { fn: vector_norm2_f32, bpe: 4 },
};

/**
 * WASM-accelerated L2 norm (Euclidean norm).
 * Returns sqrt(sum(x^2)) as a number, or null if WASM can't handle.
 */
export function wasmVectorNorm2(a: ArrayStorage): number | null {
  if (!a.isCContiguous) return null;
  if (isComplexDType(a.dtype)) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  // Float16: convert to f32 and use f32 kernel
  if (dtype === 'float16') {
    const aData = f16ToF32Input(a.data.subarray(a.offset, a.offset + size) as TypedArray, dtype);
    const aPtr = scratchCopyIn(aData);
    return vector_norm2_f32(aPtr, size);
  }

  const entry = kernels[dtype];
  if (!entry) {
    wasmConfig.wasmCallCount--; // undo increment
    return null;
  }

  const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, entry.bpe);
  return entry.fn(aPtr, size);
}
