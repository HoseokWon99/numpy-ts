/**
 * WASM-accelerated np.indices() for 2D and 3D grids.
 */

import { indices_2d, indices_3d } from './bins/indices.wasm';
import { wasmMalloc, resetScratchAllocator } from './runtime';
import { ArrayStorage } from '../storage';
import type { TypedArray } from '../dtype';
import { wasmConfig } from './config';

const BASE_THRESHOLD = 64;

/**
 * WASM-accelerated indices for 2D/3D grids with int32 dtype.
 * Returns null if WASM can't handle this case.
 */
export function wasmIndices(dimensions: number[], dtype: string): ArrayStorage | null {
  if (dtype !== 'int32') return null;

  const ndim = dimensions.length;
  if (ndim < 2 || ndim > 3) return null;

  const gridSize = dimensions.reduce((a, b) => a * b, 1);
  if (gridSize < BASE_THRESHOLD * wasmConfig.thresholdMultiplier) return null;

  const totalSize = ndim * gridSize;
  const outBytes = totalSize * 4; // i32

  const outRegion = wasmMalloc(outBytes);
  if (!outRegion) return null;

  wasmConfig.wasmCallCount++;
  resetScratchAllocator();

  if (ndim === 2) {
    indices_2d(outRegion.ptr, dimensions[0]!, dimensions[1]!);
  } else {
    indices_3d(outRegion.ptr, dimensions[0]!, dimensions[1]!, dimensions[2]!);
  }

  return ArrayStorage.fromWasmRegion(
    [ndim, ...dimensions],
    'int32',
    outRegion,
    totalSize,
    Int32Array as unknown as new (
      buffer: ArrayBuffer,
      byteOffset: number,
      length: number
    ) => TypedArray
  );
}
