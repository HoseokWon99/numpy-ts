/**
 * WASM-accelerated element-wise heaviside step function.
 *
 * Scalar: out[i] = x1[i] < 0 ? 0 : x1[i] == 0 ? x2 : 1
 * Binary: out[i] = x1[i] < 0 ? 0 : x1[i] == 0 ? x2[i] : 1
 * Returns null if WASM can't handle this case.
 */

import {
  heaviside_scalar_f64,
  heaviside_scalar_f32,
  heaviside_f64,
  heaviside_f32,
} from './bins/heaviside.wasm';
import { ensureMemory, resetAllocator, copyIn, alloc, copyOut } from './runtime';
import { ArrayStorage } from '../storage';
import type { DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type ScalarFn = (x1Ptr: number, outPtr: number, N: number, x2: number) => void;
type BinaryFn = (x1Ptr: number, x2Ptr: number, outPtr: number, N: number) => void;

const scalarKernels: Partial<Record<DType, ScalarFn>> = {
  float64: heaviside_scalar_f64,
  float32: heaviside_scalar_f32,
};

const binaryKernels: Partial<Record<DType, BinaryFn>> = {
  float64: heaviside_f64,
  float32: heaviside_f32,
};

type AnyTypedArrayCtor = new (length: number) => TypedArray;
const ctorMap: Partial<Record<DType, AnyTypedArrayCtor>> = {
  float64: Float64Array,
  float32: Float32Array,
};

export function wasmHeavisideScalar(
  x1: ArrayStorage,
  x2: number,
  resultDtype: 'float64' | 'float32'
): ArrayStorage | null {
  if (!x1.isCContiguous) return null;
  const size = x1.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const kernel = scalarKernels[resultDtype];
  const Ctor = ctorMap[resultDtype];
  if (!kernel || !Ctor) return null;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  ensureMemory(size * bpe * 2);
  resetAllocator();

  const x1Ptr = copyIn(x1.data.subarray(x1.offset, x1.offset + size) as TypedArray);
  const outPtr = alloc(size * bpe);
  kernel(x1Ptr, outPtr, size, x2);

  const outData = copyOut(
    outPtr,
    size,
    Ctor as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );
  return ArrayStorage.fromData(outData, Array.from(x1.shape), resultDtype);
}

export function wasmHeaviside(
  x1: ArrayStorage,
  x2: ArrayStorage,
  resultDtype: 'float64' | 'float32'
): ArrayStorage | null {
  if (!x1.isCContiguous || !x2.isCContiguous) return null;
  const size = x1.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const kernel = binaryKernels[resultDtype];
  const Ctor = ctorMap[resultDtype];
  if (!kernel || !Ctor) return null;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  ensureMemory(size * bpe * 3);
  resetAllocator();

  const x1Ptr = copyIn(x1.data.subarray(x1.offset, x1.offset + size) as TypedArray);
  const x2Ptr = copyIn(x2.data.subarray(x2.offset, x2.offset + size) as TypedArray);
  const outPtr = alloc(size * bpe);
  kernel(x1Ptr, x2Ptr, outPtr, size);

  const outData = copyOut(
    outPtr,
    size,
    Ctor as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );
  return ArrayStorage.fromData(outData, Array.from(x1.shape), resultDtype);
}
