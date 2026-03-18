/**
 * Python NumPy validation tests for reduction operations.
 *
 * Tests all reduction functions across:
 * - Both WASM modes (default thresholds + forced WASM threshold=0)
 * - All numeric dtypes (float64, float32, int64..int8, uint64..uint8)
 * - All axis combinations (undefined, 0, 1 for 2D; 0, 1, 2 for 3D)
 */

import { describe, it, expect, beforeAll, afterEach } from 'vitest';
import * as np from '../../src/full/index';
import { wasmConfig } from '../../src';
import { runNumPy, arraysClose, checkNumPyAvailable, getPythonInfo } from './numpy-oracle';

/** NumPy dtype string for a given numpy-ts dtype */
const NP_DTYPE: Record<string, string> = {
  float64: 'np.float64',
  float32: 'np.float32',
  int64: 'np.int64',
  int32: 'np.int32',
  int16: 'np.int16',
  int8: 'np.int8',
  uint64: 'np.uint64',
  uint32: 'np.uint32',
  uint16: 'np.uint16',
  uint8: 'np.uint8',
};

const ALL_DTYPES = Object.keys(NP_DTYPE);
const FLOAT_DTYPES = ['float64', 'float32'];
const INT_DTYPES = ['int64', 'int32', 'int16', 'int8', 'uint64', 'uint32', 'uint16', 'uint8'];

const WASM_MODES = [
  { name: 'default thresholds', multiplier: 1 },
  { name: 'forced WASM (threshold=0)', multiplier: 0 },
] as const;

/** Small 2x3 test array with values that fit in all dtypes */
const SMALL_DATA = [
  [1, 2, 3],
  [4, 5, 6],
];

/** Helper: create array and run numpy comparison */
function compareReduction(
  op: string,
  data: number[][],
  dtype: string,
  axis: number | undefined,
  npOp: string = op,
  tolerance?: number
) {
  const a = np.array(data, dtype as any);
  const npDtype = NP_DTYPE[dtype]!;
  const npAxisArg = axis !== undefined ? `, axis=${axis}` : '';

  // Call JS function
  let jsResult: any;
  if (op === 'sum') jsResult = axis !== undefined ? a.sum(axis) : a.sum();
  else if (op === 'mean') jsResult = axis !== undefined ? np.mean(a, axis) : np.mean(a);
  else if (op === 'max') jsResult = axis !== undefined ? np.max(a, axis) : np.max(a);
  else if (op === 'min') jsResult = axis !== undefined ? np.min(a, axis) : np.min(a);
  else if (op === 'prod') jsResult = axis !== undefined ? np.prod(a, axis) : np.prod(a);
  else if (op === 'argmin') jsResult = axis !== undefined ? np.argmin(a, axis) : np.argmin(a);
  else if (op === 'argmax') jsResult = axis !== undefined ? np.argmax(a, axis) : np.argmax(a);
  else if (op === 'var') jsResult = axis !== undefined ? np.var_(a, axis) : np.var_(a);
  else if (op === 'std') jsResult = axis !== undefined ? np.std(a, axis) : np.std(a);
  else if (op === 'all') jsResult = axis !== undefined ? np.all(a, axis) : np.all(a);
  else if (op === 'any') jsResult = axis !== undefined ? np.any(a, axis) : np.any(a);
  else if (op === 'nansum') jsResult = axis !== undefined ? np.nansum(a, axis) : np.nansum(a);
  else if (op === 'nanmean') jsResult = axis !== undefined ? np.nanmean(a, axis) : np.nanmean(a);
  else if (op === 'nanmin') jsResult = axis !== undefined ? np.nanmin(a, axis) : np.nanmin(a);
  else if (op === 'nanmax') jsResult = axis !== undefined ? np.nanmax(a, axis) : np.nanmax(a);
  else throw new Error(`Unknown op: ${op}`);

  // Call NumPy
  const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data)}, dtype=${npDtype})
result = np.${npOp}(a${npAxisArg})
`);

  // Compare
  if (
    typeof jsResult === 'number' ||
    typeof jsResult === 'boolean' ||
    typeof jsResult === 'bigint'
  ) {
    const jsNum = Number(jsResult);
    const npNum = typeof pyResult.value === 'number' ? pyResult.value : Number(pyResult.value);
    const tol = tolerance ?? 1e-6;
    expect(Math.abs(jsNum - npNum)).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(npNum)));
  } else {
    // Array result
    const jsArr = jsResult.toArray();
    expect(arraysClose(jsArr, pyResult.value, tolerance)).toBe(true);
  }
}

for (const mode of WASM_MODES) {
  describe(`NumPy Validation: Reductions [${mode.name}]`, () => {
    beforeAll(() => {
      wasmConfig.thresholdMultiplier = mode.multiplier;
      if (!checkNumPyAvailable()) {
        throw new Error('Python NumPy not available');
      }
      if (mode.multiplier === 1) {
        const info = getPythonInfo();
        console.log(`\n  Using Python ${info.python} with NumPy ${info.numpy} (${info.command})\n`);
      }
    });

    afterEach(() => {
      wasmConfig.thresholdMultiplier = mode.multiplier;
    });

    // ============================================================
    // Core reductions: sum, mean, max, min, prod
    // ============================================================
    for (const op of ['sum', 'mean', 'max', 'min', 'prod'] as const) {
      describe(`${op}()`, () => {
        for (const dtype of ALL_DTYPES) {
          for (const axis of [undefined, 0, 1] as const) {
            const axisLabel = axis !== undefined ? `axis=${axis}` : 'no axis';
            it(`${dtype} ${axisLabel}`, () => {
              compareReduction(op, SMALL_DATA, dtype, axis);
            });
          }
        }
      });
    }

    // ============================================================
    // argmin, argmax (no BigInt overflow issues, all dtypes)
    // ============================================================
    for (const op of ['argmin', 'argmax'] as const) {
      describe(`${op}()`, () => {
        for (const dtype of ALL_DTYPES) {
          for (const axis of [undefined, 0, 1] as const) {
            const axisLabel = axis !== undefined ? `axis=${axis}` : 'no axis';
            it(`${dtype} ${axisLabel}`, () => {
              compareReduction(op, SMALL_DATA, dtype, axis);
            });
          }
        }
      });
    }

    // ============================================================
    // var, std (returns float64, all dtypes)
    // ============================================================
    for (const op of ['var', 'std'] as const) {
      describe(`${op}()`, () => {
        for (const dtype of ALL_DTYPES) {
          for (const axis of [undefined, 0, 1] as const) {
            const axisLabel = axis !== undefined ? `axis=${axis}` : 'no axis';
            it(`${dtype} ${axisLabel}`, () => {
              compareReduction(op, SMALL_DATA, dtype, axis, op === 'var' ? 'var' : 'std', 1e-5);
            });
          }
        }
      });
    }

    // ============================================================
    // all, any (boolean reductions, all dtypes)
    // ============================================================
    for (const op of ['all', 'any'] as const) {
      describe(`${op}()`, () => {
        // Use data with a zero so all() and any() have different results
        const dataWithZero = [
          [0, 1, 2],
          [3, 4, 5],
        ];
        for (const dtype of ALL_DTYPES) {
          for (const axis of [undefined, 0, 1] as const) {
            const axisLabel = axis !== undefined ? `axis=${axis}` : 'no axis';
            it(`${dtype} ${axisLabel}`, () => {
              compareReduction(op, dataWithZero, dtype, axis);
            });
          }
        }
      });
    }

    // ============================================================
    // nan* reductions (float-only for NaN semantics, int routes to non-nan)
    // ============================================================
    for (const op of ['nansum', 'nanmean', 'nanmin', 'nanmax'] as const) {
      describe(`${op}()`, () => {
        // Test float dtypes (can have NaN)
        for (const dtype of FLOAT_DTYPES) {
          for (const axis of [undefined, 0, 1] as const) {
            const axisLabel = axis !== undefined ? `axis=${axis}` : 'no axis';
            it(`${dtype} ${axisLabel}`, () => {
              compareReduction(op, SMALL_DATA, dtype, axis, op, 1e-5);
            });
          }
        }
        // Test int dtypes (route to non-nan, should still work)
        for (const dtype of INT_DTYPES) {
          it(`${dtype} no axis (routes to non-nan)`, () => {
            compareReduction(op, SMALL_DATA, dtype, undefined, op);
          });
        }
      });
    }

    // ============================================================
    // 3D array: test axis=0, 1, 2
    // ============================================================
    describe('3D reductions', () => {
      const data3d = [
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ];

      for (const op of ['sum', 'mean', 'max', 'min'] as const) {
        for (const axis of [0, 1, 2] as const) {
          it(`${op} axis=${axis} float64`, () => {
            const a = np.array(data3d);
            let jsResult: any;
            if (op === 'sum') jsResult = a.sum(axis);
            else if (op === 'mean') jsResult = np.mean(a, axis);
            else if (op === 'max') jsResult = np.max(a, axis);
            else if (op === 'min') jsResult = np.min(a, axis);

            const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data3d)}, dtype=np.float64)
result = np.${op}(a, axis=${axis})
`);
            expect(arraysClose((jsResult as any).toArray(), pyResult.value)).toBe(true);
          });
        }
      }
    });
  });
}
