/**
 * WASM-accelerated element-wise GCD (greatest common divisor).
 *
 * Scalar: out[i] = gcd(a[i], scalar)
 * Binary: out[i] = gcd(a[i], b[i])
 * Returns null if WASM can't handle this case.
 */

import { gcd_scalar_i32, gcd_i32 } from './bins/gcd.wasm';
import { ensureMemory, resetAllocator, copyIn, alloc, copyOut } from './runtime';
import { ArrayStorage } from '../storage';
import type { TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

export function wasmGcdScalar(a: ArrayStorage, scalar: number): ArrayStorage | null {
  if (!a.isCContiguous) return null;
  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  // Only handle i32 and smaller integer types (data fits in i32)
  const dtype = a.dtype;
  if (
    dtype !== 'int32' &&
    dtype !== 'int16' &&
    dtype !== 'int8' &&
    dtype !== 'uint16' &&
    dtype !== 'uint8'
  )
    return null;

  const bpe = 4; // i32
  ensureMemory(size * bpe * 2);
  resetAllocator();

  // Convert input to i32 for the kernel
  const srcData = a.data.subarray(a.offset, a.offset + size);
  const i32Data = new Int32Array(size);
  for (let i = 0; i < size; i++) i32Data[i] = Number(srcData[i]!);

  const aPtr = copyIn(i32Data as unknown as TypedArray);
  const outPtr = alloc(size * bpe);
  gcd_scalar_i32(aPtr, outPtr, size, Math.abs(Math.trunc(scalar)));

  const outData = copyOut(
    outPtr,
    size,
    Int32Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );
  return ArrayStorage.fromData(outData, Array.from(a.shape), 'int32');
}

export function wasmGcd(a: ArrayStorage, b: ArrayStorage): ArrayStorage | null {
  if (!a.isCContiguous || !b.isCContiguous) return null;
  const size = a.size;
  if (size < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  // Only handle i32 and smaller integer types
  const aDtype = a.dtype;
  const bDtype = b.dtype;
  const validTypes = ['int32', 'int16', 'int8', 'uint16', 'uint8'];
  if (!validTypes.includes(aDtype) || !validTypes.includes(bDtype)) return null;

  const bpe = 4; // i32
  ensureMemory(size * bpe * 3);
  resetAllocator();

  // Convert both inputs to i32
  const aSrc = a.data.subarray(a.offset, a.offset + size);
  const bSrc = b.data.subarray(b.offset, b.offset + size);
  const aI32 = new Int32Array(size);
  const bI32 = new Int32Array(size);
  for (let i = 0; i < size; i++) {
    aI32[i] = Number(aSrc[i]!);
    bI32[i] = Number(bSrc[i]!);
  }

  const aPtr = copyIn(aI32 as unknown as TypedArray);
  const bPtr = copyIn(bI32 as unknown as TypedArray);
  const outPtr = alloc(size * bpe);
  gcd_i32(aPtr, bPtr, outPtr, size);

  const outData = copyOut(
    outPtr,
    size,
    Int32Array as unknown as new (buf: ArrayBuffer, off: number, len: number) => TypedArray
  );
  return ArrayStorage.fromData(outData, Array.from(a.shape), 'int32');
}
