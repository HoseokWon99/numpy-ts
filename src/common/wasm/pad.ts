/**
 * WASM-accelerated 2D constant zero-padding.
 *
 * Pads a [rows x cols] matrix with `pad_width` zeros on all sides.
 * Returns null if WASM can't handle this case.
 */

import {
  pad_2d_f64,
  pad_2d_f32,
  pad_2d_i64,
  pad_2d_i32,
  pad_2d_i16,
  pad_2d_i8,
} from './bins/pad.wasm';
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

type Pad2DFn = (aPtr: number, outPtr: number, rows: number, cols: number, padWidth: number) => void;

const kernels: Partial<Record<DType, Pad2DFn>> = {
  float64: pad_2d_f64,
  float32: pad_2d_f32,
  int64: pad_2d_i64,
  uint64: pad_2d_i64,
  int32: pad_2d_i32,
  uint32: pad_2d_i32,
  int16: pad_2d_i16,
  uint16: pad_2d_i16,
  int8: pad_2d_i8,
  uint8: pad_2d_i8,
  float16: pad_2d_f32,
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
 * WASM-accelerated 2D constant zero-pad with uniform pad_width.
 * Only handles 2D C-contiguous arrays with constant_values=0 and uniform pad_width.
 * Returns null if WASM can't handle.
 */
export function wasmPad2D(a: ArrayStorage, padWidth: number): ArrayStorage | null {
  if (!a.isCContiguous) return null;
  if (a.shape.length !== 2) return null;

  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const dtype = a.dtype;
  const kernel = kernels[dtype];
  const Ctor = ctorMap[dtype];
  if (!kernel || !Ctor) return null;

  const rows = a.shape[0]!;
  const cols = a.shape[1]!;
  const outRows = rows + 2 * padWidth;
  const outCols = cols + 2 * padWidth;
  const outSize = outRows * outCols;

  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const outBytes = outSize * bpe;
  const isF16 = dtype === 'float16';

  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  if (isF16) {
    let aData = a.data.subarray(a.offset, a.offset + size) as TypedArray;
    aData = f16ToF32Input(aData, dtype);
    const aPtr = scratchCopyIn(aData);
    kernel(aPtr, outRegion.ptr, rows, cols, padWidth);
    const mem = getSharedMemory();
    const f32View = new Float32Array(mem.buffer, outRegion.ptr, outSize);
    const f32Copy = new Float32Array(outSize);
    f32Copy.set(f32View);
    outRegion.release();
    return ArrayStorage.fromData(
      f32ToF16Output(f32Copy as unknown as TypedArray, dtype),
      [outRows, outCols],
      dtype
    );
  }

  const aPtr = resolveInputPtr(a.data, a.isWasmBacked, a.wasmPtr, a.offset, size, bpe);
  kernel(aPtr, outRegion.ptr, rows, cols, padWidth);

  return ArrayStorage.fromWasmRegion(
    [outRows, outCols],
    dtype,
    outRegion,
    outSize,
    Ctor as unknown as new (buffer: ArrayBuffer, byteOffset: number, length: number) => TypedArray
  );
}
