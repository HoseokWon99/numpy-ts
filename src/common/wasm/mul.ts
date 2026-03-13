/**
 * WASM-accelerated element-wise multiplication.
 *
 * Binary: out[i] = a[i] * b[i]  (same-shape contiguous arrays)
 * Scalar: out[i] = a[i] * scalar
 * Returns null if WASM can't handle this case.
 */

import {
  mul_f64,
  mul_f32,
  mul_i64,
  mul_i32,
  mul_i16,
  mul_i8,
  mul_c128,
  mul_c64,
  mul_scalar_f64,
  mul_scalar_f32,
  mul_scalar_i64,
  mul_scalar_i32,
  mul_scalar_i16,
  mul_scalar_i8,
  mul_scalar_c128,
  mul_scalar_c64,
} from './bins/mul.wasm';
import { ensureMemory, resetAllocator, copyIn, alloc, copyOut } from './runtime';
import { ArrayStorage } from '../storage';
import { promoteDTypes, type DType, type TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type BinaryFn = (aPtr: number, bPtr: number, outPtr: number, N: number) => void;
type ScalarFn = (aPtr: number, outPtr: number, N: number, scalar: number) => void;

const binaryKernels: Partial<Record<DType, BinaryFn>> = {
  float64: mul_f64,
  float32: mul_f32,
  int64: mul_i64,
  uint64: mul_i64,
  int32: mul_i32,
  uint32: mul_i32,
  int16: mul_i16,
  uint16: mul_i16,
  int8: mul_i8,
  uint8: mul_i8,
  complex128: mul_c128,
  complex64: mul_c64,
};

const scalarKernels: Partial<Record<DType, ScalarFn>> = {
  float64: mul_scalar_f64,
  float32: mul_scalar_f32,
  int64: mul_scalar_i64,
  uint64: mul_scalar_i64,
  int32: mul_scalar_i32,
  uint32: mul_scalar_i32,
  int16: mul_scalar_i16,
  uint16: mul_scalar_i16,
  int8: mul_scalar_i8,
  uint8: mul_scalar_i8,
  complex128: mul_scalar_c128,
  complex64: mul_scalar_c64,
};

type AnyTypedArrayCtor = new (length: number) => TypedArray;
const ctorMap: Partial<Record<DType, AnyTypedArrayCtor>> = {
  float64: Float64Array,
  float32: Float32Array,
  complex128: Float64Array,
  complex64: Float32Array,
  int64: BigInt64Array,
  uint64: BigUint64Array,
  int32: Int32Array,
  uint32: Uint32Array,
  int16: Int16Array,
  uint16: Uint16Array,
  int8: Int8Array,
  uint8: Uint8Array,
};

const complexFactor: Partial<Record<DType, number>> = {
  complex128: 2,
  complex64: 2,
};

/**
 * WASM-accelerated element-wise multiply of two same-shape contiguous arrays.
 * Returns null if WASM can't handle.
 */
export function wasmMul(a: ArrayStorage, b: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous || !b.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = promoteDTypes(a.dtype, b.dtype);
  const kernel = binaryKernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const factor = complexFactor[dtype] ?? 1;
  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const totalElements = size * factor;
  const aBytes = totalElements * bpe;
  const bBytes = totalElements * bpe;
  const outBytes = totalElements * bpe;

  ensureMemory(aBytes + bBytes + outBytes);
  resetAllocator();

  const aOff = a.offset * factor;
  const bOff = b.offset * factor;
  const aData = a.data.subarray(aOff, aOff + totalElements) as TypedArray;
  const bData = b.data.subarray(bOff, bOff + totalElements) as TypedArray;

  const aPtr = copyIn(aData);
  const bPtr = copyIn(bData);
  const outPtr = alloc(outBytes);

  kernel(aPtr, bPtr, outPtr, size);

  const outData = copyOut(
    outPtr,
    totalElements,
    Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  );

  return ArrayStorage.fromData(outData, Array.from(a.shape), dtype);
}

/**
 * WASM-accelerated element-wise multiply scalar.
 * Returns null if WASM can't handle.
 */
export function wasmMulScalar(a: ArrayStorage, scalar: number): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  const kernel = scalarKernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const factor = complexFactor[dtype] ?? 1;
  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const totalElements = size * factor;
  const aBytes = totalElements * bpe;
  const outBytes = totalElements * bpe;

  ensureMemory(aBytes + outBytes);
  resetAllocator();

  const aOff = a.offset * factor;
  const aData = a.data.subarray(aOff, aOff + totalElements) as TypedArray;

  const aPtr = copyIn(aData);
  const outPtr = alloc(outBytes);

  kernel(aPtr, outPtr, size, scalar);

  const outData = copyOut(
    outPtr,
    totalElements,
    Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  );

  return ArrayStorage.fromData(outData, Array.from(a.shape), dtype);
}
