/**
 * WASM-accelerated element-wise maximum.
 *
 * Binary: out[i] = max(a[i], b[i])
 * Scalar: out[i] = max(a[i], scalar)
 * Returns null if WASM can't handle this case.
 */

import {
  max_f64,
  max_f32,
  max_i64,
  max_i32,
  max_i16,
  max_i8,
  max_scalar_f64,
  max_scalar_f32,
  max_scalar_i64,
  max_scalar_i32,
  max_scalar_i16,
  max_scalar_i8,
  max_u64,
  max_u32,
  max_u16,
  max_u8,
  max_scalar_u64,
  max_scalar_u32,
  max_scalar_u16,
  max_scalar_u8,
} from './bins/max.wasm';
import { ensureMemory, resetAllocator, copyIn, alloc, copyOut } from './runtime';
import { ArrayStorage } from '../storage';
import { promoteDTypes, type DType, type TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type BinaryFn = (aPtr: number, bPtr: number, outPtr: number, N: number) => void;
type ScalarFn = (aPtr: number, outPtr: number, N: number, scalar: number) => void;

const binaryKernels: Partial<Record<DType, BinaryFn>> = {
  float64: max_f64,
  float32: max_f32,
  int64: max_i64,
  uint64: max_u64,
  int32: max_i32,
  uint32: max_u32,
  int16: max_i16,
  uint16: max_u16,
  int8: max_i8,
  uint8: max_u8,
};

const scalarKernels: Partial<Record<DType, ScalarFn>> = {
  float64: max_scalar_f64,
  float32: max_scalar_f32,
  int64: max_scalar_i64,
  uint64: max_scalar_u64,
  int32: max_scalar_i32,
  uint32: max_scalar_u32,
  int16: max_scalar_i16,
  uint16: max_scalar_u16,
  int8: max_scalar_i8,
  uint8: max_scalar_u8,
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

export function wasmMax(a: ArrayStorage, b: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous || !b.isCContiguous) return null;
  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = promoteDTypes(a.dtype, b.dtype);
  const kernel = binaryKernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  ensureMemory(size * bpe * 3);
  resetAllocator();

  const aPtr = copyIn(a.data.subarray(a.offset, a.offset + size) as TypedArray);
  const bPtr = copyIn(b.data.subarray(b.offset, b.offset + size) as TypedArray);
  const outPtr = alloc(size * bpe);
  kernel(aPtr, bPtr, outPtr, size);

  const outData = copyOut(
    outPtr,
    size,
    Ctor as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );
  return ArrayStorage.fromData(outData, Array.from(a.shape), dtype);
}

export function wasmMaxScalar(a: ArrayStorage, scalar: number): ArrayStorage | null {
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
  kernel(aPtr, outPtr, size, scalar);

  const outData = copyOut(
    outPtr,
    size,
    Ctor as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );
  return ArrayStorage.fromData(outData, Array.from(a.shape), dtype);
}
