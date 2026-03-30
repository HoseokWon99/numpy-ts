/**
 * WASM-accelerated flat array roll (circular shift).
 *
 * roll: out[i] = a[(i - shift + N) % N]
 * Returns null if WASM can't handle this case.
 */

import { roll_f64, roll_f32, roll_i64, roll_i32, roll_i16, roll_i8 } from './bins/roll.wasm';
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
import type { DType, TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

type RollFn = (aPtr: number, outPtr: number, N: number, shift: number) => void;

const kernels: Partial<Record<DType, RollFn>> = {
  float64: roll_f64,
  float32: roll_f32,
  int64: roll_i64,
  uint64: roll_i64,
  int32: roll_i32,
  uint32: roll_i32,
  int16: roll_i16,
  uint16: roll_i16,
  int8: roll_i8,
  uint8: roll_i8,
  float16: roll_f32,
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
  float16: Float32Array,
};

/**
 * WASM-accelerated flat roll (no axis).
 * Returns null if WASM can't handle.
 */
export function wasmRoll(a: ArrayStorage, shift: number): ArrayStorage | null {
  if (!a.isCContiguous) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  const kernel = kernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const outBytes = size * bpe;
  const isF16 = dtype === 'float16';

  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  if (isF16) {
    let aData = a.data.subarray(a.offset, a.offset + size) as TypedArray;
    aData = f16ToF32Input(aData, dtype);
    const aPtr = scratchCopyIn(aData);
    kernel(aPtr, outRegion.ptr, size, shift);
    const mem = getSharedMemory();
    const f32View = new Float32Array(mem.buffer, outRegion.ptr, size);
    const f32Copy = new Float32Array(size);
    f32Copy.set(f32View);
    outRegion.release();
    return ArrayStorage.fromData(
      f32ToF16Output(f32Copy as unknown as TypedArray, dtype),
      Array.from(a.shape),
      dtype
    );
  }

  const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);
  kernel(aPtr, outRegion.ptr, size, shift);

  return ArrayStorage.fromWasmRegion(
    Array.from(a.shape),
    dtype,
    outRegion,
    size,
    Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  );
}
