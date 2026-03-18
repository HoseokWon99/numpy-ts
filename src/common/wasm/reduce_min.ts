/**
 * WASM-accelerated reduction min.
 *
 * Reduction: result = min(a[0..N])
 * Returns null if WASM can't handle this case.
 * Unsigned types use SEPARATE kernels.
 */

import {
  reduce_min_f32,
  reduce_min_i64,
  reduce_min_i32,
  reduce_min_i16,
  reduce_min_i8,
  reduce_min_u64,
  reduce_min_u32,
  reduce_min_u16,
  reduce_min_u8,
  reduce_min_strided_f32,
  reduce_min_strided_i64,
  reduce_min_strided_i32,
  reduce_min_strided_i16,
  reduce_min_strided_i8,
  reduce_min_strided_u64,
  reduce_min_strided_u32,
  reduce_min_strided_u16,
  reduce_min_strided_u8,
} from './bins/reduce_min.wasm';
import { ensureMemory, resetAllocator, copyIn, alloc, copyOut } from './runtime';
import { ArrayStorage } from '../storage';
import type { DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type ReduceFn = (aPtr: number, N: number) => number | bigint;

const kernels: Partial<Record<DType, ReduceFn>> = {
  // float64 excluded: V2f64 SIMD (2-wide) is slower than V8's JIT'd scalar loop
  float32: reduce_min_f32,
  int64: reduce_min_i64,
  uint64: reduce_min_u64,
  int32: reduce_min_i32,
  uint32: reduce_min_u32,
  int16: reduce_min_i16,
  uint16: reduce_min_u16,
  int8: reduce_min_i8,
  uint8: reduce_min_u8,
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

/**
 * WASM-accelerated reduction min (no axis, full array).
 * Returns null if WASM can't handle (complex types, non-contiguous, too small).
 */
export function wasmReduceMin(a: ArrayStorage): number | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  const kernel = kernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;

  ensureMemory(size * bpe);
  resetAllocator();

  const aOff = a.offset;
  const aData = a.data.subarray(aOff, aOff + size) as TypedArray;
  const aPtr = copyIn(aData);

  return Number(kernel(aPtr, size));
}

// --- Strided axis reduction (output dtype matches input dtype) ---

type StridedFn = (aPtr: number, outPtr: number, outer: number, axis: number, inner: number) => void;

const stridedKernels: Partial<Record<DType, StridedFn>> = {
  float32: reduce_min_strided_f32,
  int64: reduce_min_strided_i64,
  uint64: reduce_min_strided_u64,
  int32: reduce_min_strided_i32,
  uint32: reduce_min_strided_u32,
  int16: reduce_min_strided_i16,
  uint16: reduce_min_strided_u16,
  int8: reduce_min_strided_i8,
  uint8: reduce_min_strided_u8,
};

// Output Ctor matches input dtype (min doesn't promote)
const outCtorMap: Partial<
  Record<DType, new (buf: ArrayBuffer, off: number, len: number) => TypedArray>
> = {
  float32: Float32Array as unknown as new (
    buf: ArrayBuffer,
    off: number,
    len: number
  ) => TypedArray,
  int64: BigInt64Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray,
  uint64: BigUint64Array as unknown as new (
    buf: ArrayBuffer,
    off: number,
    len: number
  ) => TypedArray,
  int32: Int32Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray,
  uint32: Uint32Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray,
  int16: Int16Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray,
  uint16: Uint16Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray,
  int8: Int8Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray,
  uint8: Uint8Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray,
};

/**
 * WASM-accelerated strided min along an axis.
 * Output dtype matches input dtype. Returns output ArrayStorage, or null if WASM can't handle.
 */
export function wasmReduceMinStrided(
  a: ArrayStorage,
  outerSize: number,
  axisSize: number,
  innerSize: number
): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const totalSize = outerSize * axisSize * innerSize;
  if (totalSize < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  const kernel = stridedKernels[dtype];
  const InCtor = ctorMap[dtype];
  const OutCtor = outCtorMap[dtype];
  if (!kernel || !InCtor || !OutCtor) return null;

  const inBpe = (InCtor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const outBpe = inBpe; // output type matches input type for min
  const outSize = outerSize * innerSize;

  ensureMemory(totalSize * inBpe + outSize * outBpe);
  resetAllocator();

  const aOff = a.offset;
  const aData = a.data.subarray(aOff, aOff + totalSize) as TypedArray;
  const inPtr = copyIn(aData);
  const outPtr = alloc(outSize * outBpe);

  kernel(inPtr, outPtr, outerSize, axisSize, innerSize);

  const outData = copyOut(outPtr, outSize, OutCtor);

  return ArrayStorage.fromData(outData, [outSize], dtype);
}
