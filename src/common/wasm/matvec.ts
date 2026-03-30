/**
 * WASM-accelerated matrix-vector product.
 *
 * Computes y[i] = sum_k A[i,k] * x[k] for A[M,K] and x[K].
 * Returns null if WASM can't handle this case.
 */

import {
  matvec_f64,
  matvec_f32,
  matvec_c128,
  matvec_c64,
  matvec_i64,
  matvec_i32,
  matvec_i16,
  matvec_i8,
} from './bins/matvec.wasm';
import {
  wasmMalloc,
  resetScratchAllocator,
  resolveInputPtr,
  scratchCopyIn,
  getSharedMemory,
  f16ToF32Input,
  f32ToF16Output,
} from './runtime';
import { ArrayStorage } from '../storage';
import { promoteDTypes, type DType, type TypedArray } from '../dtype';

import { wasmConfig } from './config';

const BASE_THRESHOLD = 128; // Minimum M*K for WASM

type WasmMatvecFn = (A: number, x: number, y: number, M: number, K: number) => void;

const wasmKernels: Partial<Record<DType, WasmMatvecFn>> = {
  float64: matvec_f64,
  float32: matvec_f32,
  complex128: matvec_c128,
  complex64: matvec_c64,
  int64: matvec_i64,
  uint64: matvec_i64,
  int32: matvec_i32,
  uint32: matvec_i32,
  int16: matvec_i16,
  uint16: matvec_i16,
  int8: matvec_i8,
  uint8: matvec_i8,
  float16: matvec_f32,
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
  float16: Float32Array,
};

const complexFactor: Partial<Record<DType, number>> = {
  complex128: 2,
  complex64: 2,
};

/**
 * WASM-accelerated matvec: A[M,K] · x[K] → y[M].
 * A must be 2D, x must be 1D, both contiguous.
 */
export function wasmMatvec(A: ArrayStorage, x: ArrayStorage): ArrayStorage | null {
  if (A.ndim !== 2 || x.ndim !== 1) return null;
  if (!A.isCContiguous || !x.isCContiguous) return null;

  const M = A.shape[0]!;
  const K = A.shape[1]!;
  if (K !== x.shape[0]!) return null;
  if (M * K < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const resultDtype = promoteDTypes(A.dtype, x.dtype);
  const kernel = wasmKernels[resultDtype];
  const Ctor = ctorMap[resultDtype];
  if (!kernel || !Ctor) return null;

  const factor = complexFactor[resultDtype] ?? 1;
  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const totalElements = M * factor;
  const outBytes = totalElements * bpe;
  const isF16 = resultDtype === 'float16';

  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  if (isF16) {
    let aData = A.data.subarray(
      A.offset * factor,
      A.offset * factor + M * K * factor
    ) as TypedArray;
    let xData = x.data.subarray(x.offset * factor, x.offset * factor + K * factor) as TypedArray;
    aData = f16ToF32Input(aData, resultDtype);
    xData = f16ToF32Input(xData, resultDtype);
    const aPtr = scratchCopyIn(aData);
    const xPtr = scratchCopyIn(xData);
    kernel(aPtr, xPtr, outRegion.ptr, M, K);
    const mem = getSharedMemory();
    const f32View = new Float32Array(mem.buffer, outRegion.ptr, totalElements);
    const f32Copy = new Float32Array(totalElements);
    f32Copy.set(f32View);
    outRegion.release();
    return ArrayStorage.fromData(
      f32ToF16Output(f32Copy as unknown as TypedArray, resultDtype),
      [M],
      resultDtype
    );
  }

  const aPtr = resolveInputPtr(
    A.data,
    A.isWasmBacked,
    A.wasmPtr,
    A.offset * factor,
    M * K * factor,
    bpe
  );
  const xPtr = resolveInputPtr(
    x.data,
    x.isWasmBacked,
    x.wasmPtr,
    x.offset * factor,
    K * factor,
    bpe
  );

  kernel(aPtr, xPtr, outRegion.ptr, M, K);

  return ArrayStorage.fromWasmRegion(
    [M],
    resultDtype,
    outRegion,
    totalElements,
    Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  );
}
