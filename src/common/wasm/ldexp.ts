/**
 * WASM-accelerated element-wise ldexp (x1 * 2^x2).
 *
 * Scalar variant only (x2 is a single integer).
 * Returns null if WASM can't handle this case.
 */

import { ldexp_scalar_f64, ldexp_scalar_f32 } from './bins/ldexp.wasm';
import { ensureMemory, resetAllocator, copyIn, alloc, copyOut } from './runtime';
import { ArrayStorage } from '../storage';
import type { DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type ScalarFn = (x1Ptr: number, outPtr: number, N: number, exp: number) => void;

const scalarKernels: Partial<Record<DType, ScalarFn>> = {
  float64: ldexp_scalar_f64,
  float32: ldexp_scalar_f32,
};

type AnyTypedArrayCtor = new (length: number) => TypedArray;
const ctorMap: Partial<Record<DType, AnyTypedArrayCtor>> = {
  float64: Float64Array,
  float32: Float32Array,
};

export function wasmLdexpScalar(a: ArrayStorage, exp: number): ArrayStorage | null {
  if (!a.isCContiguous) return null;
  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  const kernel = scalarKernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  ensureMemory(size * bpe * 2);
  resetAllocator();

  const aPtr = copyIn(a.data.subarray(a.offset, a.offset + size) as TypedArray);
  const outPtr = alloc(size * bpe);
  kernel(aPtr, outPtr, size, exp);

  const outData = copyOut(
    outPtr,
    size,
    Ctor as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );
  return ArrayStorage.fromData(outData, Array.from(a.shape), dtype);
}
