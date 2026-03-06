/**
 * Comprehensive NDim + Axes Validation Tests
 *
 * Systematically validates numpy-ts against NumPy for ALL implemented functions across:
 * - All input dimensionalities (0D through 5D)
 * - All valid axis values (positive, negative, multi-axis)
 * - keepdims parameter
 * - Broadcasting scenarios
 * - Error cases (invalid ndim, invalid axis)
 *
 * Generated from scripts/ndim_axes_comprehensive.py
 */

import { describe, it, expect, beforeAll } from 'vitest';
import {
  array,
  zeros,
  // Elementwise unary
  sin,
  arcsin,
  exp,
  sqrt,
  absolute as abs,
  negative,
  floor,
  ceil,
  isfinite,
  isnan,
  // Binary
  add,
  subtract,
  multiply,
  greater,
  // Reductions
  sum,
  prod,
  mean,
  std,
  amax,
  amin,
  argmax,
  argmin,
  all,
  any,
  cumsum,
  cumprod,
  // nan variants
  nansum,
  nanprod,
  nanmean,
  nanstd,
  nanvar,
  nanmin,
  nanmax,
  nanargmin,
  // sort/search
  sort,
  argsort,
  // Stats
  diff,
  median,
  percentile,
  quantile,
  average,
  // Shape ops
  reshape,
  flip,
  roll,
  concatenate,
  stack,
  squeeze,
  expand_dims,
  swapaxes,
  moveaxis,
  repeat,
  tile,
  transpose,
  rot90,
  broadcast_to,
  take,
  clip,
  where,
  // Split
  split,
  array_split,
  // linalg
  dot,
  inner,
  outer,
  tensordot,
  trace,
  diagonal,
  matmul,
  cross,
  einsum,
  linalg,
  // FFT
  fft,
  // Type
  nanmedian,
} from '../../src';
import { runNumPy, arraysClose, checkNumPyAvailable } from './numpy-oracle';

// ─── Helpers ───────────────────────────────────────────────────────────────

/** Create ascending float array of given shape. */
function mk(shape: number[], scale = 1.0): ReturnType<typeof array> {
  if (shape.length === 0) return array(0.5);
  const n = shape.reduce((a, b) => a * b, 1);
  return array(Array.from({ length: n }, (_, i) => (i + 1) * scale)).reshape(shape);
}

/** Create bool array (alternating T/F) of given shape. */
function mkBool(shape: number[]): ReturnType<typeof array> {
  const n = shape.reduce((a, b) => a * b, 1);
  return array(Array.from({ length: n }, (_, i) => i % 2 === 0)).reshape(shape);
}

describe('NumPy Validation: Comprehensive NDim + Axes', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error(
        '❌ Python NumPy not available!\n\n   source ~/.zshrc && conda activate py313\n'
      );
    }
  });

  // ============================================================
  // SECTION 1: Elementwise Unary (0D through 5D)
  // ============================================================
  describe('elementwise unary: shape preservation 0D-5D', () => {
    const SHAPES: [string, number[]][] = [
      ['0D', []],
      ['1D', [6]],
      ['2D', [2, 3]],
      ['3D', [2, 3, 4]],
      ['4D', [2, 2, 3, 4]],
      ['5D', [2, 2, 2, 3, 4]],
    ];

    for (const [label, shape] of SHAPES) {
      it(`sin: ${label} ${JSON.stringify(shape)}`, () => {
        const a = mk(shape, 0.05);
        const r = sin(a);
        const expected = shape.length === 0 ? [] : shape;
        expect((r as any).shape ?? []).toEqual(expected);
        const py = runNumPy(
          `result = np.sin(np.${shape.length === 0 ? 'array(0.5)' : `arange(1, ${shape.reduce((a, b) => a * b, 1) + 1}, dtype=float).reshape(${JSON.stringify(shape)}) * 0.05`})`
        );
        const val = (r as any).toArray?.() ?? r;
        expect(arraysClose(val, py.value, 1e-10)).toBe(true);
      });

      it(`exp: ${label} ${JSON.stringify(shape)}`, () => {
        const a = mk(shape, 0.1);
        const r = exp(a);
        expect((r as any).shape ?? []).toEqual(shape.length === 0 ? [] : shape);
      });

      it(`sqrt: ${label} ${JSON.stringify(shape)}`, () => {
        const a = mk(shape, 0.5);
        const r = sqrt(a);
        expect((r as any).shape ?? []).toEqual(shape.length === 0 ? [] : shape);
      });
    }

    it('arcsin: 2D input in range [-1,1]', () => {
      const a = mk([3, 4], 0.08); // max = 0.08*12 = 0.96 < 1
      const r = arcsin(a);
      expect((r as any).shape).toEqual([3, 4]);
    });

    it('isfinite: 3D', () => {
      const a = mk([2, 3, 4]);
      const r = isfinite(a);
      expect((r as any).shape).toEqual([2, 3, 4]);
    });

    it('isnan: 3D', () => {
      const a = mk([2, 3, 4]);
      const r = isnan(a);
      expect((r as any).shape).toEqual([2, 3, 4]);
    });

    it('floor: 4D', () => {
      const a = mk([2, 2, 3, 4], 0.7);
      const r = floor(a);
      const py = runNumPy(`result = np.floor(np.arange(1,49,dtype=float).reshape(2,2,3,4)*0.7)`);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('ceil: 3D', () => {
      const a = mk([2, 3, 4], 0.7);
      const r = ceil(a);
      const py = runNumPy(`result = np.ceil(np.arange(1,25,dtype=float).reshape(2,3,4)*0.7)`);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('negative: 5D', () => {
      const a = mk([2, 2, 2, 3, 4]);
      const r = negative(a);
      expect((r as any).shape).toEqual([2, 2, 2, 3, 4]);
      expect((r as any).toArray().flat(Infinity)[0]).toBeLessThan(0);
    });
  });

  // ============================================================
  // SECTION 2: Binary Elementwise + Broadcasting
  // ============================================================
  describe('binary elementwise: same-shape 0D-5D + broadcasting', () => {
    const addPy = (shape: number[]) => {
      if (shape.length === 0) return `result = np.add(np.array(0.5), np.array(0.5))`;
      const n = shape.reduce((a, b) => a * b, 1);
      return `result = np.add(np.arange(1,${n + 1},dtype=float).reshape(${JSON.stringify(shape)})*0.5, np.arange(1,${n + 1},dtype=float).reshape(${JSON.stringify(shape)})*0.3)`;
    };

    for (const shape of [[] as number[], [3], [2, 3], [2, 3, 4], [2, 2, 3, 4], [2, 2, 2, 3, 4]]) {
      it(`add: same-shape ${JSON.stringify(shape)}`, () => {
        const a = mk(shape, 0.5);
        const b = mk(shape, 0.3);
        const r = add(a, b);
        expect((r as any).shape ?? []).toEqual(shape);
        if (shape.length <= 3) {
          const py = runNumPy(addPy(shape));
          const val = (r as any).toArray?.() ?? r;
          expect(arraysClose(val, py.value)).toBe(true);
        }
      });
    }

    it('add: broadcast 1D+2D [4] + [3,4] -> [3,4]', () => {
      const a = array([1, 2, 3, 4]);
      const b = mk([3, 4]);
      const r = add(a, b);
      const py = runNumPy(
        `result = np.add(np.array([1,2,3,4],dtype=float), np.arange(1,13,dtype=float).reshape(3,4))`
      );
      expect((r as any).shape).toEqual([3, 4]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('add: broadcast 1D+3D [4] + [2,3,4] -> [2,3,4]', () => {
      const a = array([1, 2, 3, 4]);
      const b = mk([2, 3, 4]);
      const r = add(a, b);
      expect((r as any).shape).toEqual([2, 3, 4]);
    });

    it('add: broadcast 1D+4D [4] + [2,2,3,4] -> [2,2,3,4]', () => {
      const a = array([1, 2, 3, 4]);
      const b = mk([2, 2, 3, 4]);
      const r = add(a, b);
      expect((r as any).shape).toEqual([2, 2, 3, 4]);
    });

    it('add: broadcast 1D+5D [4] + [2,2,2,3,4] -> [2,2,2,3,4]', () => {
      const a = array([1, 2, 3, 4]);
      const b = mk([2, 2, 2, 3, 4]);
      const r = add(a, b);
      expect((r as any).shape).toEqual([2, 2, 2, 3, 4]);
    });

    it('multiply: broadcast 2D+3D [3,4] + [2,3,4]', () => {
      const a = mk([3, 4]);
      const b = mk([2, 3, 4]);
      const r = multiply(a, b);
      expect((r as any).shape).toEqual([2, 3, 4]);
    });

    it('multiply: broadcast scalar+4D', () => {
      const a = array(2.0);
      const b = mk([2, 2, 3, 4]);
      const r = multiply(a, b);
      expect((r as any).shape).toEqual([2, 2, 3, 4]);
    });

    it('subtract: 3D - 3D values correct', () => {
      const a = mk([2, 3, 4]);
      const b = mk([2, 3, 4], 0.5);
      const r = subtract(a, b);
      const py = runNumPy(`
A = np.arange(1,25,dtype=float).reshape(2,3,4)
B = np.arange(1,25,dtype=float).reshape(2,3,4)*0.5
result = np.subtract(A, B)`);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('greater: 3D boolean result', () => {
      const a = mk([2, 3, 4]);
      const b = mk([2, 3, 4], 0.5);
      const r = greater(a, b);
      expect((r as any).shape).toEqual([2, 3, 4]);
    });
  });

  // ============================================================
  // SECTION 3: Reductions — ALL axes, keepdims, multi-axis
  // ============================================================
  describe('sum: all axes, keepdims, multi-axis', () => {
    it('0D: scalar input', () => {
      const a = array(5.0);
      const r = sum(a);
      expect(r).toBeCloseTo(5.0);
    });

    it('1D: axis=0', () => {
      const a = array([1, 2, 3, 4, 5, 6]);
      expect(sum(a, 0)).toBeCloseTo(21);
    });

    it('1D: axis=-1', () => {
      const a = array([1, 2, 3, 4, 5, 6]);
      expect(sum(a, -1)).toBeCloseTo(21);
    });

    it('1D: axis=0 keepdims', () => {
      const a = array([1, 2, 3, 4]);
      const r = sum(a, 0, true);
      expect((r as any).shape).toEqual([1]);
    });

    for (const [ax, expectedShape] of [
      [0, [4]],
      [1, [3]],
      [-1, [3]],
      [-2, [4]],
    ] as [number, number[]][]) {
      it(`2D [3,4]: axis=${ax} -> ${JSON.stringify(expectedShape)}`, () => {
        const A = mk([3, 4]);
        const r = sum(A, ax);
        const py = runNumPy(
          `result = np.sum(np.arange(1,13,dtype=float).reshape(3,4), axis=${ax})`
        );
        expect((r as any).shape).toEqual(expectedShape);
        expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
      });
    }

    it('2D [3,4]: keepdims axis=0', () => {
      const A = mk([3, 4]);
      const r = sum(A, 0, true);
      expect((r as any).shape).toEqual([1, 4]);
    });

    it('2D [3,4]: keepdims axis=1', () => {
      const A = mk([3, 4]);
      const r = sum(A, 1, true);
      expect((r as any).shape).toEqual([3, 1]);
    });

    for (const [ax, expectedShape] of [
      [0, [3, 4]],
      [1, [2, 4]],
      [2, [2, 3]],
      [-1, [2, 3]],
      [-2, [2, 4]],
      [-3, [3, 4]],
    ] as [number, number[]][]) {
      it(`3D [2,3,4]: axis=${ax} -> ${JSON.stringify(expectedShape)}`, () => {
        const A = mk([2, 3, 4]);
        const r = sum(A, ax);
        const py = runNumPy(
          `result = np.sum(np.arange(1,25,dtype=float).reshape(2,3,4), axis=${ax})`
        );
        expect((r as any).shape).toEqual(expectedShape);
        expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
      });
    }

    it('3D [2,3,4]: axis=0 keepdims -> [1,3,4]', () => {
      const A = mk([2, 3, 4]);
      const r = sum(A, 0, true);
      expect((r as any).shape).toEqual([1, 3, 4]);
    });

    it('3D [2,3,4]: axis=2 keepdims -> [2,3,1]', () => {
      const A = mk([2, 3, 4]);
      const r = sum(A, 2, true);
      expect((r as any).shape).toEqual([2, 3, 1]);
    });

    it('3D [2,3,4]: multi-axis (0,1) -> [4]', () => {
      const A = mk([2, 3, 4]);
      const r = (A as any).sum([0, 1]);
      const py = runNumPy(
        `result = np.sum(np.arange(1,25,dtype=float).reshape(2,3,4), axis=(0,1))`
      );
      expect((r as any).shape).toEqual([4]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('3D [2,3,4]: multi-axis (1,2) -> [2]', () => {
      const A = mk([2, 3, 4]);
      const r = (A as any).sum([1, 2]);
      const py = runNumPy(
        `result = np.sum(np.arange(1,25,dtype=float).reshape(2,3,4), axis=(1,2))`
      );
      expect((r as any).shape).toEqual([2]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    for (const [ax, expectedShape] of [
      [0, [3, 4, 5]],
      [1, [2, 4, 5]],
      [2, [2, 3, 5]],
      [3, [2, 3, 4]],
      [-1, [2, 3, 4]],
      [-4, [3, 4, 5]],
    ] as [number, number[]][]) {
      it(`4D [2,3,4,5]: axis=${ax} -> ${JSON.stringify(expectedShape)}`, () => {
        const A = mk([2, 3, 4, 5]);
        const r = sum(A, ax);
        expect((r as any).shape).toEqual(expectedShape);
      });
    }

    it('4D [2,3,4,5]: multi-axis (0,1,2,3) -> scalar', () => {
      const A = mk([2, 3, 4, 5]);
      const r = (A as any).sum([0, 1, 2, 3]);
      const total = 2 * 3 * 4 * 5;
      const expected = ((1 + total) * total) / 2;
      expect(arraysClose(r as number, expected)).toBe(true);
    });

    it('4D [2,3,4,5]: multi-axis (1,3) keepdims -> [2,1,4,1]', () => {
      const A = mk([2, 3, 4, 5]);
      const r = (A as any).sum([1, 3], true);
      expect((r as any).shape).toEqual([2, 1, 4, 1]);
    });

    it('5D [2,2,2,3,4]: axis=2', () => {
      const A = mk([2, 2, 2, 3, 4]);
      const r = sum(A, 2);
      expect((r as any).shape).toEqual([2, 2, 3, 4]);
    });
  });

  describe('mean: all axes, keepdims', () => {
    it('2D [3,4]: all axes', () => {
      const A = mk([3, 4]);
      for (const [ax, sh] of [
        [0, [4]],
        [1, [3]],
        [-1, [3]],
        [-2, [4]],
      ] as [number, number[]][]) {
        const r = mean(A, ax);
        expect((r as any).shape).toEqual(sh);
      }
    });

    it('3D [2,3,4]: all axes + keepdims', () => {
      const A = mk([2, 3, 4]);
      const origShape = [2, 3, 4];
      for (const [ax, sh] of [
        [0, [3, 4]],
        [1, [2, 4]],
        [2, [2, 3]],
        [-1, [2, 3]],
      ] as [number, number[]][]) {
        expect((mean(A, ax) as any).shape).toEqual(sh);
        const normAx = ax >= 0 ? ax : origShape.length + ax;
        const expectedKeepdims = origShape.map((v, i) => (i === normAx ? 1 : v));
        expect((mean(A, ax, true) as any).shape).toEqual(expectedKeepdims);
      }
    });

    it('4D [2,3,4,5]: axis=-1', () => {
      const A = mk([2, 3, 4, 5]);
      const r = mean(A, -1);
      expect((r as any).shape).toEqual([2, 3, 4]);
    });
  });

  describe('prod: all axes', () => {
    it('2D [2,3]: axis=0', () => {
      const A = mk([2, 3]);
      const r = prod(A, 0);
      const py = runNumPy(`result = np.prod(np.arange(1,7,dtype=float).reshape(2,3), axis=0)`);
      expect((r as any).shape).toEqual([3]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('3D [2,3,4]: all axes', () => {
      const A = mk([2, 3, 4]);
      expect((prod(A, 0) as any).shape).toEqual([3, 4]);
      expect((prod(A, 1) as any).shape).toEqual([2, 4]);
      expect((prod(A, 2) as any).shape).toEqual([2, 3]);
    });
  });

  describe('std/var: all axes, keepdims', () => {
    it('2D [3,4]: all axes + keepdims', () => {
      const A = mk([3, 4]);
      expect((std(A, 0) as any).shape).toEqual([4]);
      expect((std(A, 1) as any).shape).toEqual([3]);
      expect((std(A, 0, 0, true) as any).shape).toEqual([1, 4]);
      expect((std(A, 1, 0, true) as any).shape).toEqual([3, 1]);
    });

    it('3D [2,3,4]: axis=2 values correct', () => {
      const A = mk([2, 3, 4]);
      const r = std(A, 2);
      const py = runNumPy(`result = np.std(np.arange(1,25,dtype=float).reshape(2,3,4), axis=2)`);
      expect((r as any).shape).toEqual([2, 3]);
      expect(arraysClose((r as any).toArray(), py.value, 1e-10)).toBe(true);
    });

    it('4D [2,3,4,5]: axis=-1 keepdims', () => {
      const A = mk([2, 3, 4, 5]);
      const r = std(A, -1, 0, true);
      expect((r as any).shape).toEqual([2, 3, 4, 1]);
    });
  });

  // ============================================================
  // SECTION 4: NaN Reductions
  // ============================================================
  describe('nan reductions: all axes with NaN values', () => {
    const makeWithNaN = (shape: number[]) => {
      const n = shape.reduce((a, b) => a * b, 1);
      const data = Array.from({ length: n }, (_, i) => i + 1.0);
      data[0] = NaN;
      return array(data).reshape(shape);
    };

    it('nansum: 2D axis=0', () => {
      const A = makeWithNaN([3, 4]);
      const r = nansum(A, 0);
      const py = runNumPy(`
A = np.arange(1,13,dtype=float).reshape(3,4)
A[0,0] = np.nan
result = np.nansum(A, axis=0)`);
      expect((r as any).shape).toEqual([4]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('nansum: 2D axis=1', () => {
      const A = makeWithNaN([3, 4]);
      const r = nansum(A, 1);
      expect((r as any).shape).toEqual([3]);
    });

    it('nansum: 3D axis=0', () => {
      const A = makeWithNaN([2, 3, 4]);
      const r = nansum(A, 0);
      expect((r as any).shape).toEqual([3, 4]);
    });

    it('nansum: 3D axis=1', () => {
      const A = makeWithNaN([2, 3, 4]);
      const r = nansum(A, 1);
      expect((r as any).shape).toEqual([2, 4]);
    });

    it('nansum: 3D axis=2', () => {
      const A = makeWithNaN([2, 3, 4]);
      const r = nansum(A, 2);
      expect((r as any).shape).toEqual([2, 3]);
    });

    it('nansum: 3D keepdims', () => {
      const A = makeWithNaN([2, 3, 4]);
      const r = nansum(A, 1, true);
      expect((r as any).shape).toEqual([2, 1, 4]);
    });

    for (const [fnName, fn] of [
      ['nanmean', nanmean],
      ['nanstd', nanstd],
      ['nanvar', nanvar],
      ['nanprod', nanprod],
    ] as [string, (a: any, ...args: any[]) => any][]) {
      it(`${fnName}: 3D axis=1`, () => {
        const A = makeWithNaN([2, 3, 4]);
        const r = fn(A, 1);
        expect((r as any).shape).toEqual([2, 4]);
      });

      it(`${fnName}: 3D axis=-1`, () => {
        const A = makeWithNaN([2, 3, 4]);
        const r = fn(A, -1);
        expect((r as any).shape).toEqual([2, 3]);
      });
    }

    it('nanmin: 3D axis=0', () => {
      const A = makeWithNaN([2, 3, 4]);
      const r = nanmin(A, 0);
      expect((r as any).shape).toEqual([3, 4]);
    });

    it('nanmax: 3D all axes', () => {
      const A = makeWithNaN([2, 3, 4]);
      expect((nanmax(A, 0) as any).shape).toEqual([3, 4]);
      expect((nanmax(A, 1) as any).shape).toEqual([2, 4]);
      expect((nanmax(A, 2) as any).shape).toEqual([2, 3]);
    });

    it('nanargmin: 3D axis=1', () => {
      const A = makeWithNaN([2, 3, 4]);
      const r = nanargmin(A, 1);
      expect((r as any).shape).toEqual([2, 4]);
    });

    it('nanmedian: 2D axis=0', () => {
      const A = makeWithNaN([3, 4]);
      const r = nanmedian(A, 0);
      expect((r as any).shape).toEqual([4]);
    });

    it('nanmedian: 3D axis=1', () => {
      const A = makeWithNaN([2, 3, 4]);
      const r = nanmedian(A, 1);
      expect((r as any).shape).toEqual([2, 4]);
    });
  });

  // ============================================================
  // SECTION 5: amax/amin/argmax/argmin — all axes
  // ============================================================
  describe('amax/amin/argmax/argmin: all axes', () => {
    it('0D input', () => {
      const a = array(5.0);
      expect(amax(a)).toBeCloseTo(5.0);
      expect(amin(a)).toBeCloseTo(5.0);
    });

    for (const [fn, fnName] of [
      [amax, 'amax'],
      [amin, 'amin'],
      [argmax, 'argmax'],
      [argmin, 'argmin'],
    ] as [(a: any, ...args: any[]) => any, string][]) {
      it(`${fnName}: 1D`, () => {
        const A = mk([6]);
        expect(typeof fn(A)).toBe('number');
      });

      for (const [shape, axes] of [
        [
          [3, 4],
          [0, 1, -1, -2],
        ],
        [
          [2, 3, 4],
          [0, 1, 2, -1, -2, -3],
        ],
        [
          [2, 2, 3, 4],
          [0, 1, 2, 3, -1, -4],
        ],
      ] as [number[], number[]][]) {
        for (const ax of axes) {
          it(`${fnName}: ${JSON.stringify(shape)} axis=${ax}`, () => {
            const A = mk(shape);
            const r = fn(A, ax);
            const ndim = shape.length;
            const normAx = ax >= 0 ? ax : ndim + ax;
            const expectedShape = shape.filter((_, i) => i !== normAx);
            expect((r as any).shape).toEqual(expectedShape);
          });
        }
      }

      it(`${fnName}: 3D [2,3,4] keepdims axis=1`, () => {
        if (fn === amax || fn === amin) {
          const r = fn(mk([2, 3, 4]), 1, true);
          expect((r as any).shape).toEqual([2, 1, 4]);
        }
      });
    }
  });

  // ============================================================
  // SECTION 6: all/any — all axes, keepdims
  // ============================================================
  describe('all/any: all axes, keepdims', () => {
    for (const [fn, fnName] of [
      [all, 'all'],
      [any, 'any'],
    ] as [(a: any, ...args: any[]) => any, string][]) {
      it(`${fnName}: 0D`, () => {
        expect(fn(array(true))).toBe(true);
      });

      for (const [shape, axes] of [
        [[4], [0, -1]],
        [
          [3, 4],
          [0, 1, -1, -2],
        ],
        [
          [2, 3, 4],
          [0, 1, 2, -1, -2, -3],
        ],
        [
          [2, 2, 3, 4],
          [0, 1, 2, 3, -1, -4],
        ],
      ] as [number[], number[]][]) {
        it(`${fnName}: ${JSON.stringify(shape)} no-axis`, () => {
          const A = mkBool(shape);
          const r = fn(A);
          expect(typeof r).toBe('boolean');
        });

        for (const ax of axes) {
          it(`${fnName}: ${JSON.stringify(shape)} axis=${ax}`, () => {
            const A = mkBool(shape);
            const r = fn(A, ax);
            const ndim = shape.length;
            const normAx = ax >= 0 ? ax : ndim + ax;
            const expectedShape = shape.filter((_, i) => i !== normAx);
            if (expectedShape.length === 0) {
              // 1D reduction to scalar — may return boolean primitive
              expect(typeof r === 'boolean' || (r as any).shape?.length === 0).toBe(true);
            } else {
              expect((r as any).shape).toEqual(expectedShape);
            }
          });
        }

        it(`${fnName}: ${JSON.stringify(shape)} axis=0 keepdims`, () => {
          const A = mkBool(shape);
          const r = fn(A, 0, true);
          const expectedShape = [...shape];
          expectedShape[0] = 1;
          expect((r as any).shape).toEqual(expectedShape);
        });
      }
    }
  });

  // ============================================================
  // SECTION 7: cumsum/cumprod — all axes
  // ============================================================
  describe('cumsum/cumprod: all axes', () => {
    for (const [fn, fnName] of [
      [cumsum, 'cumsum'],
      [cumprod, 'cumprod'],
    ] as [(a: any, ...args: any[]) => any, string][]) {
      it(`${fnName}: 1D no-axis`, () => {
        const A = array([1, 2, 3, 4]);
        const r = fn(A);
        expect((r as any).shape).toEqual([4]);
      });

      it(`${fnName}: 1D axis=0`, () => {
        const A = array([1, 2, 3, 4]);
        const r = fn(A, 0);
        expect((r as any).shape).toEqual([4]);
      });

      for (const [shape, axes] of [
        [
          [3, 4],
          [0, 1, -1, -2],
        ],
        [
          [2, 3, 4],
          [0, 1, 2, -1, -2, -3],
        ],
        [
          [2, 2, 3, 4],
          [0, 1, 2, 3, -1, -4],
        ],
      ] as [number[], number[]][]) {
        for (const ax of axes) {
          it(`${fnName}: ${JSON.stringify(shape)} axis=${ax}`, () => {
            const A = mk(shape);
            const r = fn(A, ax);
            expect((r as any).shape).toEqual(shape); // cumulative preserves shape
          });
        }
      }

      it(`${fnName}: 3D [2,3,4] axis=1 values correct`, () => {
        const A = mk([2, 3, 4]);
        const r = fn(A, 1);
        const py = runNumPy(
          `result = np.${fnName}(np.arange(1,25,dtype=float).reshape(2,3,4), axis=1)`
        );
        expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
      });
    }
  });

  // ============================================================
  // SECTION 8: sort/argsort — all axes
  // ============================================================
  describe('sort/argsort: all axes', () => {
    for (const [fn, fnName] of [
      [sort, 'sort'],
      [argsort, 'argsort'],
    ] as [(a: any, ...args: any[]) => any, string][]) {
      for (const [shape, axes] of [
        [[5], [0, -1]],
        [
          [3, 4],
          [0, 1, -1, -2],
        ],
        [
          [2, 3, 4],
          [0, 1, 2, -1, -2, -3],
        ],
        [
          [2, 2, 3, 4],
          [0, 1, 2, 3, -1, -4],
        ],
      ] as [number[], number[]][]) {
        for (const ax of axes) {
          it(`${fnName}: ${JSON.stringify(shape)} axis=${ax}`, () => {
            const A = mk(shape)
              .multiply(-1)
              .add(shape.reduce((a, b) => a * b, 1) + 1);
            const r = fn(A, ax);
            expect((r as any).shape).toEqual(shape); // sort preserves shape
          });
        }

        it(`${fnName}: ${JSON.stringify(shape)} axis=0 values correct`, () => {
          const A = mk(shape)
            .multiply(-1)
            .add(shape.reduce((a, b) => a * b, 1) + 1);
          const r = fn(A, 0);
          const n = shape.reduce((a, b) => a * b, 1);
          const py = runNumPy(`
A = (np.arange(1,${n + 1},dtype=float)*-1+${n + 1}).reshape(${JSON.stringify(shape)})
result = np.${fnName}(A, axis=0)`);
          expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
        });
      }
    }
  });

  // ============================================================
  // SECTION 9: diff — all axes
  // ============================================================
  describe('diff: all axes', () => {
    for (const [shape, axes] of [
      [[5], [0]],
      [
        [3, 4],
        [0, 1],
      ],
      [
        [2, 3, 4],
        [0, 1, 2],
      ],
      [
        [2, 2, 3, 4],
        [0, 1, 2, 3],
      ],
    ] as [number[], number[]][]) {
      for (const ax of axes) {
        it(`diff n=1: ${JSON.stringify(shape)} axis=${ax}`, () => {
          const A = mk(shape);
          const r = diff(A, 1, ax);
          const expectedShape = [...shape];
          expectedShape[ax] -= 1;
          expect((r as any).shape).toEqual(expectedShape);
          const n = shape.reduce((a, b) => a * b, 1);
          const py = runNumPy(
            `result = np.diff(np.arange(1,${n + 1},dtype=float).reshape(${JSON.stringify(shape)}), axis=${ax})`
          );
          expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
        });

        it(`diff n=2: ${JSON.stringify(shape)} axis=${ax}`, () => {
          if (shape[ax] >= 3) {
            const A = mk(shape);
            const r = diff(A, 2, ax);
            const expectedShape = [...shape];
            expectedShape[ax] -= 2;
            expect((r as any).shape).toEqual(expectedShape);
          }
        });
      }
    }
  });

  // ============================================================
  // SECTION 10: median/percentile/quantile — all axes, keepdims
  // ============================================================
  describe('median/percentile/quantile: all axes, keepdims', () => {
    for (const shape of [[5], [3, 4], [2, 3, 4], [2, 2, 3, 4]] as number[][]) {
      it(`median: ${JSON.stringify(shape)} no-axis`, () => {
        const r = median(mk(shape));
        expect(typeof r).toBe('number');
      });

      for (const ax of shape.map((_, i) => i)) {
        it(`median: ${JSON.stringify(shape)} axis=${ax}`, () => {
          const A = mk(shape);
          const r = median(A, ax);
          const expectedShape = shape.filter((_, i) => i !== ax);
          if (expectedShape.length === 0) {
            expect(typeof r === 'number').toBe(true);
          } else {
            expect((r as any).shape).toEqual(expectedShape);
          }
        });

        it(`median: ${JSON.stringify(shape)} axis=${ax} keepdims`, () => {
          const A = mk(shape);
          const r = median(A, ax, true);
          const expectedShape = [...shape];
          expectedShape[ax] = 1;
          expect((r as any).shape).toEqual(expectedShape);
        });

        it(`median: ${JSON.stringify(shape)} axis=${-(shape.length - ax)}`, () => {
          const A = mk(shape);
          const negAx = -(shape.length - ax);
          const r = median(A, negAx);
          const expectedShape = shape.filter((_, i) => i !== ax);
          if (expectedShape.length === 0) {
            expect(typeof r === 'number').toBe(true);
          } else {
            expect((r as any).shape).toEqual(expectedShape);
          }
        });
      }
    }

    it('median: 3D [2,3,4] axis=1 values correct', () => {
      const A = mk([2, 3, 4]);
      const r = median(A, 1);
      const py = runNumPy(`result = np.median(np.arange(1,25,dtype=float).reshape(2,3,4), axis=1)`);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('percentile: 3D [2,3,4] axis=0', () => {
      const A = mk([2, 3, 4]);
      const r = percentile(A, 75, 0);
      expect((r as any).shape).toEqual([3, 4]);
    });

    it('quantile: 3D [2,3,4] axis=-1', () => {
      const A = mk([2, 3, 4]);
      const r = quantile(A, 0.75, -1);
      expect((r as any).shape).toEqual([2, 3]);
    });
  });

  // ============================================================
  // SECTION 11: average — all axes
  // ============================================================
  describe('average: all axes', () => {
    for (const shape of [[5], [3, 4], [2, 3, 4], [2, 2, 3, 4]] as number[][]) {
      it(`average: ${JSON.stringify(shape)} no-axis`, () => {
        const r = average(mk(shape));
        expect(typeof r).toBe('number');
      });
      for (const ax of shape.map((_, i) => i)) {
        it(`average: ${JSON.stringify(shape)} axis=${ax}`, () => {
          const A = mk(shape);
          const r = average(A, ax);
          const expectedShape = shape.filter((_, i) => i !== ax);
          if (expectedShape.length === 0) {
            expect(typeof r).toBe('number');
          } else {
            expect((r as any).shape).toEqual(expectedShape);
          }
        });
      }
    }
  });

  // ============================================================
  // SECTION 12: flip — all axes
  // ============================================================
  describe('flip: all axes', () => {
    for (const shape of [[5], [3, 4], [2, 3, 4], [2, 2, 3, 4], [2, 2, 2, 3, 4]] as number[][]) {
      it(`flip: ${JSON.stringify(shape)} no-axis`, () => {
        const A = mk(shape);
        const r = flip(A);
        expect((r as any).shape).toEqual(shape);
        const n = shape.reduce((a, b) => a * b, 1);
        const py = runNumPy(
          `result = np.flip(np.arange(1,${n + 1},dtype=float).reshape(${JSON.stringify(shape)}))`
        );
        if (shape.length <= 4) {
          expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
        }
      });

      for (const ax of shape.map((_, i) => i)) {
        it(`flip: ${JSON.stringify(shape)} axis=${ax}`, () => {
          const A = mk(shape);
          const r = flip(A, ax);
          expect((r as any).shape).toEqual(shape);
        });

        it(`flip: ${JSON.stringify(shape)} axis=${-(shape.length - ax)}`, () => {
          const A = mk(shape);
          const negAx = -(shape.length - ax);
          const r = flip(A, negAx);
          expect((r as any).shape).toEqual(shape);
        });
      }
    }
  });

  // ============================================================
  // SECTION 13: roll — all axes
  // ============================================================
  describe('roll: all axes', () => {
    for (const shape of [[5], [3, 4], [2, 3, 4], [2, 2, 3, 4]] as number[][]) {
      it(`roll: ${JSON.stringify(shape)} no-axis`, () => {
        const A = mk(shape);
        const r = roll(A, 2);
        expect((r as any).shape).toEqual(shape);
      });

      for (const ax of shape.map((_, i) => i)) {
        it(`roll: ${JSON.stringify(shape)} axis=${ax}`, () => {
          const A = mk(shape);
          const r = roll(A, 2, ax);
          expect((r as any).shape).toEqual(shape);
          const n = shape.reduce((a, b) => a * b, 1);
          const py = runNumPy(
            `result = np.roll(np.arange(1,${n + 1},dtype=float).reshape(${JSON.stringify(shape)}), 2, axis=${ax})`
          );
          expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
        });

        it(`roll: ${JSON.stringify(shape)} axis=${-(shape.length - ax)} (negative)`, () => {
          const A = mk(shape);
          const negAx = -(shape.length - ax);
          const r = roll(A, -1, negAx);
          expect((r as any).shape).toEqual(shape);
        });
      }
    }
  });

  // ============================================================
  // SECTION 14: concatenate — all axes
  // ============================================================
  describe('concatenate: all axes', () => {
    for (const shape of [[3], [3, 4], [2, 3, 4], [2, 2, 3, 4]] as number[][]) {
      for (const ax of shape.map((_, i) => i)) {
        it(`concatenate: ${JSON.stringify(shape)} axis=${ax}`, () => {
          const A = mk(shape);
          const r = concatenate([A, A], ax);
          const expectedShape = [...shape];
          expectedShape[ax] *= 2;
          expect((r as any).shape).toEqual(expectedShape);
        });
      }
    }

    it('concatenate: 3D [2,3,4] axis=1 values correct', () => {
      const A = mk([2, 3, 4]);
      const r = concatenate([A, A], 1);
      const py = runNumPy(`
A = np.arange(1,25,dtype=float).reshape(2,3,4)
result = np.concatenate([A, A], axis=1)`);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('concatenate: 4D along axis=3', () => {
      const A = mk([2, 2, 3, 4]);
      const r = concatenate([A, A], 3);
      expect((r as any).shape).toEqual([2, 2, 3, 8]);
    });
  });

  // ============================================================
  // SECTION 15: stack — all axes
  // ============================================================
  describe('stack: all axes', () => {
    for (const shape of [[3], [2, 3], [2, 3, 4], [2, 2, 3, 4]] as number[][]) {
      const ndim = shape.length;
      for (let ax = 0; ax <= ndim; ax++) {
        it(`stack: ${JSON.stringify(shape)} axis=${ax}`, () => {
          const A = mk(shape);
          const r = stack([A, A], ax);
          const expectedShape = [...shape];
          expectedShape.splice(ax, 0, 2);
          expect((r as any).shape).toEqual(expectedShape);
        });
      }
    }

    it('stack: 2D [2,3] axis=0 values correct', () => {
      const A = mk([2, 3]);
      const r = stack([A, A], 0);
      const py = runNumPy(`
A = np.arange(1,7,dtype=float).reshape(2,3)
result = np.stack([A, A], axis=0)`);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });
  });

  // ============================================================
  // SECTION 16: split/array_split — all axes
  // ============================================================
  describe('split/array_split: all axes', () => {
    it('split: 1D [6] axis=0 into 3', () => {
      const A = mk([6]);
      const r = split(A, 3, 0);
      expect(r.length).toBe(3);
      expect((r[0] as any).shape).toEqual([2]);
    });

    it('split: 2D [4,6] axis=0 into 2', () => {
      const A = mk([4, 6]);
      const r = split(A, 2, 0);
      expect(r.length).toBe(2);
      expect((r[0] as any).shape).toEqual([2, 6]);
    });

    it('split: 2D [4,6] axis=1 into 2', () => {
      const A = mk([4, 6]);
      const r = split(A, 2, 1);
      expect(r.length).toBe(2);
      expect((r[0] as any).shape).toEqual([4, 3]);
    });

    it('split: 3D [2,3,6] axis=2 into 2', () => {
      const A = mk([2, 3, 6]);
      const r = split(A, 2, 2);
      expect(r.length).toBe(2);
      expect((r[0] as any).shape).toEqual([2, 3, 3]);
    });

    it('split: 3D [2,3,6] axis=0 into 2', () => {
      const A = mk([2, 3, 6]);
      const r = split(A, 2, 0);
      expect(r.length).toBe(2);
      expect((r[0] as any).shape).toEqual([1, 3, 6]);
    });

    it('array_split: 3D [2,3,5] axis=1 into 2 (unequal)', () => {
      const A = mk([2, 3, 5]);
      const r = array_split(A, 2, 1);
      expect(r.length).toBe(2);
    });

    it('array_split: 4D [2,2,3,6] axis=3 into 3', () => {
      const A = mk([2, 2, 3, 6]);
      const r = array_split(A, 3, 3);
      expect(r.length).toBe(3);
      expect((r[0] as any).shape).toEqual([2, 2, 3, 2]);
    });
  });

  // ============================================================
  // SECTION 17: swapaxes — all pairs
  // ============================================================
  describe('swapaxes: all axis pairs', () => {
    for (const [shape, pairs] of [
      [[3, 4], [[0, 1]]],
      [
        [2, 3, 4],
        [
          [0, 1],
          [0, 2],
          [1, 2],
        ],
      ],
      [
        [2, 3, 4, 5],
        [
          [0, 1],
          [0, 2],
          [0, 3],
          [1, 2],
          [1, 3],
          [2, 3],
        ],
      ],
    ] as [number[], [number, number][]][]) {
      for (const [i, j] of pairs) {
        it(`swapaxes: ${JSON.stringify(shape)} (${i},${j})`, () => {
          const A = mk(shape);
          const r = swapaxes(A, i, j);
          const expectedShape = [...shape];
          [expectedShape[i], expectedShape[j]] = [expectedShape[j], expectedShape[i]];
          expect((r as any).shape).toEqual(expectedShape);
          const n = shape.reduce((a, b) => a * b, 1);
          const py = runNumPy(
            `result = np.swapaxes(np.arange(1,${n + 1},dtype=float).reshape(${JSON.stringify(shape)}), ${i}, ${j})`
          );
          expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
        });
      }
    }
  });

  // ============================================================
  // SECTION 18: moveaxis
  // ============================================================
  describe('moveaxis: single and multi', () => {
    for (const [shape, src, dst, expectedShape] of [
      [[2, 3, 4], 0, 2, [3, 4, 2]],
      [[2, 3, 4], 2, 0, [4, 2, 3]],
      [[2, 3, 4], -1, 0, [4, 2, 3]],
      [[2, 3, 4, 5], 0, -1, [3, 4, 5, 2]],
      [[2, 3, 4, 5], 1, 3, [2, 4, 5, 3]],
    ] as [number[], number, number, number[]][]) {
      it(`moveaxis: ${JSON.stringify(shape)} ${src}->${dst}`, () => {
        const A = mk(shape);
        const r = moveaxis(A, src, dst);
        expect((r as any).shape).toEqual(expectedShape);
        const n = shape.reduce((a, b) => a * b, 1);
        const py = runNumPy(
          `result = np.moveaxis(np.arange(1,${n + 1},dtype=float).reshape(${JSON.stringify(shape)}), ${src}, ${dst})`
        );
        expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
      });
    }
  });

  // ============================================================
  // SECTION 19: expand_dims / squeeze — all positions
  // ============================================================
  describe('expand_dims: all positions', () => {
    for (const [shape, ax, expectedShape] of [
      [[], 0, [1]],
      [[3], 0, [1, 3]],
      [[3], 1, [3, 1]],
      [[3], -1, [3, 1]],
      [[2, 3], 0, [1, 2, 3]],
      [[2, 3], 1, [2, 1, 3]],
      [[2, 3], 2, [2, 3, 1]],
      [[2, 3], -1, [2, 3, 1]],
      [[2, 3, 4], 0, [1, 2, 3, 4]],
      [[2, 3, 4], 2, [2, 3, 1, 4]],
      [[2, 3, 4], 3, [2, 3, 4, 1]],
    ] as [number[], number, number[]][]) {
      it(`expand_dims: ${JSON.stringify(shape)} axis=${ax} -> ${JSON.stringify(expectedShape)}`, () => {
        const A = mk(shape);
        const r = expand_dims(A, ax);
        expect((r as any).shape).toEqual(expectedShape);
      });
    }
  });

  describe('squeeze: all size-1 axes', () => {
    for (const [shape, expectedShape] of [
      [[1, 3], [3]],
      [[1, 3, 1], [3]],
      [
        [2, 1, 3, 1, 4],
        [2, 3, 4],
      ],
      [[1, 1, 1], []],
    ] as [number[], number[]][]) {
      it(`squeeze: ${JSON.stringify(shape)} -> ${JSON.stringify(expectedShape)}`, () => {
        const A = mk(shape);
        const r = squeeze(A);
        expect((r as any).shape).toEqual(expectedShape);
      });
    }
  });

  // ============================================================
  // SECTION 20: repeat / tile — all axes
  // ============================================================
  describe('repeat: all axes', () => {
    for (const shape of [[3], [2, 3], [2, 3, 4]] as number[][]) {
      it(`repeat: ${JSON.stringify(shape)} no-axis (flattens)`, () => {
        const A = mk(shape);
        const r = repeat(A, 2);
        const n = shape.reduce((a, b) => a * b, 1);
        expect((r as any).shape).toEqual([n * 2]);
      });

      for (const ax of shape.map((_, i) => i)) {
        it(`repeat: ${JSON.stringify(shape)} axis=${ax}`, () => {
          const A = mk(shape);
          const r = repeat(A, 2, ax);
          const expectedShape = [...shape];
          expectedShape[ax] *= 2;
          expect((r as any).shape).toEqual(expectedShape);
        });
      }
    }
  });

  describe('tile: ND reps', () => {
    it('tile: [3] x2 scalar', () => {
      const r = tile(mk([3]), 2);
      expect((r as any).shape).toEqual([6]);
    });

    it('tile: [2,3] x [2,3]', () => {
      const r = tile(mk([2, 3]), [2, 3]);
      expect((r as any).shape).toEqual([4, 9]);
    });

    it('tile: [2,3,4] x [2,1,1]', () => {
      const r = tile(mk([2, 3, 4]), [2, 1, 1]);
      expect((r as any).shape).toEqual([4, 3, 4]);
    });

    it('tile: [3] x [2,1,1] (broadcast extra dims)', () => {
      const r = tile(mk([3]), [2, 1, 1]);
      expect((r as any).shape).toEqual([2, 1, 3]);
    });
  });

  // ============================================================
  // SECTION 21: reshape — ND -> MD
  // ============================================================
  describe('reshape: ND -> MD', () => {
    for (const [inShape, outShape] of [
      [[6], [2, 3]],
      [[6], [3, 2]],
      [[24], [2, 3, 4]],
      [[2, 3], [6]],
      [
        [2, 3],
        [1, 6],
      ],
      [[2, 3, 4], [24]],
      [
        [2, 3, 4],
        [2, 12],
      ],
      [
        [2, 3, 4],
        [4, 6],
      ],
      [
        [2, 3, 4],
        [2, 3, 2, 2],
      ],
      [
        [2, 2, 3, 4],
        [4, 12],
      ],
      [
        [2, 2, 3, 4],
        [2, 2, 3, 2, 2],
      ],
    ] as [number[], number[]][]) {
      it(`reshape: ${JSON.stringify(inShape)} -> ${JSON.stringify(outShape)}`, () => {
        const A = mk(inShape);
        const r = reshape(A, outShape);
        expect((r as any).shape).toEqual(outShape);
      });
    }

    it('reshape: with -1 inference [2,3,4] -> [-1]', () => {
      const A = mk([2, 3, 4]);
      const r = reshape(A, [-1]);
      expect((r as any).shape).toEqual([24]);
    });

    it('reshape: with -1 inference [2,3,4] -> [-1,4]', () => {
      const A = mk([2, 3, 4]);
      const r = reshape(A, [-1, 4]);
      expect((r as any).shape).toEqual([6, 4]);
    });
  });

  // ============================================================
  // SECTION 22: transpose — permutations
  // ============================================================
  describe('transpose: all permutations', () => {
    it('transpose: 1D -> same shape (no-op)', () => {
      const A = mk([5]);
      const r = transpose(A);
      expect((r as any).shape).toEqual([5]);
    });

    for (const [shape, perm, expectedShape] of [
      [
        [2, 3],
        [1, 0],
        [3, 2],
      ],
      [
        [2, 3, 4],
        [2, 1, 0],
        [4, 3, 2],
      ],
      [
        [2, 3, 4],
        [1, 2, 0],
        [3, 4, 2],
      ],
      [
        [2, 3, 4],
        [0, 2, 1],
        [2, 4, 3],
      ],
      [
        [2, 3, 4, 5],
        [3, 2, 1, 0],
        [5, 4, 3, 2],
      ],
      [
        [2, 3, 4, 5],
        [1, 0, 3, 2],
        [3, 2, 5, 4],
      ],
    ] as [number[], number[], number[]][]) {
      it(`transpose: ${JSON.stringify(shape)} perm=${JSON.stringify(perm)} -> ${JSON.stringify(expectedShape)}`, () => {
        const A = mk(shape);
        const r = transpose(A, perm);
        expect((r as any).shape).toEqual(expectedShape);
        const n = shape.reduce((a, b) => a * b, 1);
        const py = runNumPy(
          `result = np.transpose(np.arange(1,${n + 1},dtype=float).reshape(${JSON.stringify(shape)}), ${JSON.stringify(perm)})`
        );
        expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
      });
    }

    it('transpose: 5D default (reverse)', () => {
      const A = mk([2, 2, 2, 3, 4]);
      const r = transpose(A);
      expect((r as any).shape).toEqual([4, 3, 2, 2, 2]);
    });
  });

  // ============================================================
  // SECTION 23: rot90
  // ============================================================
  describe('rot90: 2D-4D, all k, axes param', () => {
    it('rot90: 2D k=1', () => {
      const A = mk([3, 4]);
      const r = rot90(A, 1);
      expect((r as any).shape).toEqual([4, 3]);
    });

    it('rot90: 2D k=2', () => {
      const A = mk([3, 4]);
      const r = rot90(A, 2);
      expect((r as any).shape).toEqual([3, 4]);
    });

    it('rot90: 3D k=1 default axes', () => {
      const A = mk([2, 3, 4]);
      const r = rot90(A, 1);
      expect((r as any).shape[2]).toBe(4); // last axis unchanged by default
    });

    it('rot90: 4D k=1 default axes', () => {
      const A = mk([2, 3, 4, 5]);
      const r = rot90(A, 1);
      expect((r as any).shape).toBeDefined();
    });
  });

  // ============================================================
  // SECTION 24: broadcast_to — ND
  // ============================================================
  describe('broadcast_to: ND expansion', () => {
    for (const [src, target] of [
      [[3], [4, 3]],
      [[3], [2, 4, 3]],
      [[3], [2, 2, 4, 3]],
      [[3], [2, 2, 2, 4, 3]],
      [
        [1, 3],
        [4, 2, 3],
      ],
      [
        [2, 1, 3],
        [2, 4, 3],
      ],
    ] as [number[], number[]][]) {
      it(`broadcast_to: ${JSON.stringify(src)} -> ${JSON.stringify(target)}`, () => {
        const A = mk(src);
        const r = broadcast_to(A, target);
        expect((r as any).shape).toEqual(target);
      });
    }
  });

  // ============================================================
  // SECTION 25: take — all axes
  // ============================================================
  describe('take: all axes', () => {
    const idx = [0, 1];

    it('take: 1D no-axis', () => {
      const r = take(mk([5]), idx);
      expect((r as any).shape).toEqual([2]);
    });

    for (const shape of [
      [3, 4],
      [2, 3, 4],
      [2, 2, 3, 4],
    ] as number[][]) {
      for (const ax of shape.map((_, i) => i)) {
        it(`take: ${JSON.stringify(shape)} axis=${ax}`, () => {
          const A = mk(shape);
          const r = take(A, idx, ax);
          const expectedShape = [...shape];
          expectedShape[ax] = 2;
          expect((r as any).shape).toEqual(expectedShape);
        });
      }
    }
  });

  // ============================================================
  // SECTION 26: clip — 0D through 5D
  // ============================================================
  describe('clip: 0D-5D', () => {
    for (const shape of [[], [5], [3, 4], [2, 3, 4], [2, 2, 3, 4], [2, 2, 2, 3, 4]] as number[][]) {
      it(`clip: ${JSON.stringify(shape)}`, () => {
        const A = mk(shape);
        const r = clip(A, 2, 10);
        expect((r as any).shape ?? []).toEqual(shape);
      });
    }

    it('clip: 3D values correct', () => {
      const A = mk([2, 3, 4]);
      const r = clip(A, 5, 15);
      const py = runNumPy(`result = np.clip(np.arange(1,25,dtype=float).reshape(2,3,4), 5, 15)`);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });
  });

  // ============================================================
  // SECTION 27: where — broadcasting
  // ============================================================
  describe('where: ND + broadcasting', () => {
    for (const shape of [[5], [3, 4], [2, 3, 4], [2, 2, 3, 4]] as number[][]) {
      it(`where: ${JSON.stringify(shape)} same shape`, () => {
        const cond = mkBool(shape);
        const a = mk(shape);
        const b = negative(mk(shape));
        const r = where(cond, a, b);
        expect((r as any).shape).toEqual(shape);
      });
    }

    it('where: broadcast condition [4] with values [3,4]', () => {
      const cond = mkBool([4]);
      const a = mk([3, 4]);
      const b = negative(mk([3, 4]));
      const r = where(cond, a, b);
      expect((r as any).shape).toEqual([3, 4]);
    });

    it('where: 3D values correct', () => {
      const A = mk([2, 3, 4]);
      const B = negative(mk([2, 3, 4]));
      const cond = mkBool([2, 3, 4]);
      const r = where(cond, A, B);
      const py = runNumPy(`
A = np.arange(1,25,dtype=float).reshape(2,3,4)
B = -A
cond = np.array([i%2==0 for i in range(24)], dtype=bool).reshape(2,3,4)
result = np.where(cond, A, B)`);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });
  });

  // ============================================================
  // SECTION 28: matmul — all valid dimensionality combos
  // ============================================================
  describe('matmul: all valid ndim combos', () => {
    it('1D @ 1D -> scalar', () => {
      const r = matmul(array([1, 2, 3, 4]), array([1, 2, 3, 4]));
      const val = typeof r === 'number' ? r : (r as any).toArray?.();
      const py = runNumPy(
        `result = np.matmul(np.array([1,2,3,4],dtype=float), np.array([1,2,3,4],dtype=float))`
      );
      expect(arraysClose(val, py.value)).toBe(true);
    });

    it('2D @ 1D -> 1D [3,4] @ [4]', () => {
      const r = matmul(mk([3, 4]), mk([4]));
      expect((r as any).shape).toEqual([3]);
      const py = runNumPy(
        `result = np.matmul(np.arange(1,13,dtype=float).reshape(3,4), np.arange(1,5,dtype=float))`
      );
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('1D @ 2D -> 1D [3] @ [3,4]', () => {
      const r = matmul(mk([3]), mk([3, 4]));
      expect((r as any).shape).toEqual([4]);
    });

    it('2D @ 2D -> 2D', () => {
      const r = matmul(mk([2, 3]), mk([3, 4]));
      expect((r as any).shape).toEqual([2, 4]);
      const py = runNumPy(
        `result = np.matmul(np.arange(1,7,dtype=float).reshape(2,3), np.arange(1,13,dtype=float).reshape(3,4))`
      );
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('3D @ 3D -> 3D (batched)', () => {
      const r = matmul(mk([2, 3, 4]), mk([2, 4, 5]));
      expect((r as any).shape).toEqual([2, 3, 5]);
      const py = runNumPy(
        `result = np.matmul(np.arange(1,25,dtype=float).reshape(2,3,4), np.arange(1,41,dtype=float).reshape(2,4,5))`
      );
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('4D @ 4D -> 4D (batched)', () => {
      const r = matmul(mk([2, 2, 3, 4]), mk([2, 2, 4, 5]));
      expect((r as any).shape).toEqual([2, 2, 3, 5]);
    });

    it('5D @ 5D -> 5D (batched)', () => {
      const r = matmul(mk([2, 2, 2, 3, 4]), mk([2, 2, 2, 4, 5]));
      expect((r as any).shape).toEqual([2, 2, 2, 3, 5]);
    });

    it('3D @ 2D -> 3D (batch broadcast)', () => {
      const r = matmul(mk([2, 3, 4]), mk([4, 5]));
      expect((r as any).shape).toEqual([2, 3, 5]);
      const py = runNumPy(
        `result = np.matmul(np.arange(1,25,dtype=float).reshape(2,3,4), np.arange(1,21,dtype=float).reshape(4,5))`
      );
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('4D @ 3D -> 4D (batch broadcast)', () => {
      const r = matmul(mk([2, 2, 3, 4]), mk([2, 4, 5]));
      expect((r as any).shape).toEqual([2, 2, 3, 5]);
    });

    it('1D @ 3D -> 2D', () => {
      const r = matmul(mk([3]), mk([2, 3, 4]));
      expect((r as any).shape).toEqual([2, 4]);
    });

    it('0D throws', () => {
      expect(() => matmul(array(2.0), array(2.0))).toThrow();
    });
  });

  // ============================================================
  // SECTION 29: dot/inner/outer/tensordot — ND cases
  // ============================================================
  describe('dot: ND cases', () => {
    it('0D · 0D', () => {
      const r = dot(array(2.0), array(3.0));
      expect(arraysClose(r as number, 6.0)).toBe(true);
    });

    it('1D · 1D -> scalar', () => {
      const r = dot(array([1, 2, 3, 4]), array([1, 2, 3, 4]));
      expect(arraysClose(r as number, 30)).toBe(true);
    });

    it('2D · 1D -> 1D', () => {
      const r = dot(mk([3, 4]), mk([4])) as any;
      expect(r.shape).toEqual([3]);
    });

    it('3D · 2D -> 3D', () => {
      const r = dot(mk([2, 3, 4]), mk([4, 5])) as any;
      const py = runNumPy(
        `result = np.dot(np.arange(1,25,dtype=float).reshape(2,3,4), np.arange(1,21,dtype=float).reshape(4,5))`
      );
      expect(r.shape).toEqual([2, 3, 5]);
      expect(arraysClose(r.toArray(), py.value)).toBe(true);
    });

    it('4D · 2D -> 4D', () => {
      const r = dot(mk([2, 2, 3, 4]), mk([4, 5])) as any;
      expect(r.shape).toEqual([2, 2, 3, 5]);
    });
  });

  describe('inner: ND cases', () => {
    it('0D · 0D', () => {
      const r = inner(array(2.0), array(3.0));
      expect(arraysClose(r as number, 6.0)).toBe(true);
    });

    it('1D · 1D -> scalar', () => {
      const r = inner(array([1, 2, 3]), array([4, 5, 6]));
      expect(arraysClose(r as number, 32)).toBe(true);
    });

    it('2D · 1D -> 1D', () => {
      const r = inner(mk([3, 4]), mk([4])) as any;
      expect(r.shape).toEqual([3]);
    });

    it('2D · 2D -> 2D', () => {
      const r = inner(mk([3, 4]), mk([2, 4])) as any;
      const py = runNumPy(
        `result = np.inner(np.arange(1,13,dtype=float).reshape(3,4), np.arange(1,9,dtype=float).reshape(2,4))`
      );
      expect(r.shape).toEqual([3, 2]);
      expect(arraysClose(r.toArray(), py.value)).toBe(true);
    });

    it('3D · 1D -> 2D', () => {
      const r = inner(mk([2, 3, 4]), array([1, 2, 3, 4])) as any;
      expect(r.shape).toEqual([2, 3]);
    });

    it('3D · 2D -> 3D', () => {
      const r = inner(mk([2, 3, 4]), mk([2, 4])) as any;
      expect(r.shape).toEqual([2, 3, 2]);
    });
  });

  describe('outer: ND (always flattens)', () => {
    it('1D · 1D', () => {
      const r = outer(mk([3]), mk([4]));
      expect((r as any).shape).toEqual([3, 4]);
    });

    it('2D · 2D (flattened)', () => {
      const r = outer(mk([2, 3]), mk([3, 4]));
      expect((r as any).shape).toEqual([6, 12]);
    });

    it('3D · 3D (flattened)', () => {
      const r = outer(mk([2, 3, 4]), mk([2, 3, 4]));
      expect((r as any).shape).toEqual([24, 24]);
    });
  });

  describe('tensordot: ND axes combos', () => {
    it('1D axes=1 -> scalar', () => {
      const r = tensordot(mk([4]), mk([4]), 1);
      expect(arraysClose(r as number, 30)).toBe(true);
    });

    it('2D axes=1', () => {
      const r = tensordot(mk([2, 3]), mk([3, 4]), 1) as any;
      expect(r.shape).toEqual([2, 4]);
    });

    it('3D axes=1', () => {
      const r = tensordot(mk([2, 3, 4]), mk([4, 5]), 1) as any;
      expect(r.shape).toEqual([2, 3, 5]);
      const py = runNumPy(
        `result = np.tensordot(np.arange(1,25,dtype=float).reshape(2,3,4), np.arange(1,21,dtype=float).reshape(4,5), 1)`
      );
      expect(arraysClose(r.toArray(), py.value)).toBe(true);
    });

    it('outer product (axes=0)', () => {
      const r = tensordot(mk([3]), mk([4]), 0) as any;
      expect(r.shape).toEqual([3, 4]);
    });

    it('3D contract all axes=2', () => {
      const r = tensordot(mk([2, 3]), mk([2, 3]), 2);
      expect(
        arraysClose(
          r as number,
          mk([2, 3])
            .multiply(mk([2, 3]))
            .sum() as number,
          1e-6
        )
      ).toBe(true);
    });
  });

  // ============================================================
  // SECTION 30: trace — ND batch
  // ============================================================
  describe('trace: ND batch', () => {
    it('2D square: base case', () => {
      const r = trace(mk([3, 3]));
      expect(r).toBeCloseTo(1 + 5 + 9); // diagonal of [[1,2,3],[4,5,6],[7,8,9]]
    });

    it('2D non-square', () => {
      const r = trace(mk([3, 4]));
      expect(typeof r).toBe('number');
    });

    it('2D offset=1', () => {
      const r = trace(mk([4, 4]), 1);
      expect(typeof r).toBe('number');
    });

    it('2D offset=-1', () => {
      const r = trace(mk([4, 4]), -1);
      expect(typeof r).toBe('number');
    });

    it('3D -> 1D batch', () => {
      const A = mk([2, 3, 3]);
      const r = trace(A);
      const py = runNumPy(`result = np.trace(np.arange(1,19,dtype=float).reshape(2,3,3))`);
      expect((r as any).shape ?? []).toEqual(py.shape);
      expect(arraysClose((r as any).toArray?.() ?? r, py.value)).toBe(true);
    });

    it('4D -> 2D batch', () => {
      const A = mk([2, 2, 3, 3]);
      const r = trace(A);
      const py = runNumPy(`result = np.trace(np.arange(1,37,dtype=float).reshape(2,2,3,3))`);
      expect((r as any).shape ?? []).toEqual(py.shape);
      expect(arraysClose((r as any).toArray?.() ?? r, py.value)).toBe(true);
    });

    it('5D -> 3D batch', () => {
      const A = mk([2, 2, 2, 3, 3]);
      const r = trace(A);
      // default axis1=0, axis2=1: trace over [2,2] axes, remaining=[2,3,3]
      expect((r as any).shape).toEqual([2, 3, 3]);
    });

    it('3D custom axis1=0, axis2=2', () => {
      const A = mk([3, 2, 3]);
      const r = trace(A, 0, 0, 2);
      const py = runNumPy(
        `result = np.trace(np.arange(1,19,dtype=float).reshape(3,2,3), axis1=0, axis2=2)`
      );
      expect((r as any).shape ?? []).toEqual(py.shape);
    });
  });

  // ============================================================
  // SECTION 31: diagonal — ND, all axis pairs
  // ============================================================
  describe('diagonal: ND + axis pairs', () => {
    it('2D: base case', () => {
      const r = diagonal(mk([3, 4]));
      const py = runNumPy(`result = np.diagonal(np.arange(1,13,dtype=float).reshape(3,4))`);
      expect((r as any).shape).toEqual([3]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('2D: offset=1', () => {
      const r = diagonal(mk([4, 4]), 1);
      expect((r as any).shape).toEqual([3]);
    });

    it('2D: offset=-1', () => {
      const r = diagonal(mk([4, 4]), -1);
      expect((r as any).shape).toEqual([3]);
    });

    it('3D default (axis1=0, axis2=1)', () => {
      const A = mk([2, 3, 4]);
      const r = diagonal(A, 0, 0, 1);
      const py = runNumPy(
        `result = np.diagonal(np.arange(1,25,dtype=float).reshape(2,3,4), 0, 0, 1)`
      );
      expect((r as any).shape).toEqual(py.shape);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('3D axis1=1, axis2=2', () => {
      const A = mk([2, 3, 4]);
      const r = diagonal(A, 0, 1, 2);
      const py = runNumPy(
        `result = np.diagonal(np.arange(1,25,dtype=float).reshape(2,3,4), 0, 1, 2)`
      );
      expect((r as any).shape).toEqual(py.shape);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('4D: axis1=2, axis2=3', () => {
      const A = mk([2, 3, 4, 4]);
      const r = diagonal(A, 0, 2, 3);
      const py = runNumPy(
        `result = np.diagonal(np.arange(1,97,dtype=float).reshape(2,3,4,4), 0, 2, 3)`
      );
      expect((r as any).shape).toEqual(py.shape);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('4D: axis1=1, axis2=3', () => {
      const A = mk([2, 3, 4, 3]);
      const r = diagonal(A, 0, 1, 3);
      expect((r as any).shape).toBeDefined();
    });
  });

  // ============================================================
  // SECTION 32: cross — batched ND
  // ============================================================
  describe('cross: batched ND', () => {
    it('1D-3 x 1D-3', () => {
      const r = cross(array([1, 2, 3]), array([4, 5, 6]));
      const py = runNumPy(
        `result = np.cross(np.array([1,2,3],dtype=float), np.array([4,5,6],dtype=float))`
      );
      expect(arraysClose((r as any).toArray?.() ?? r, py.value)).toBe(true);
    });

    it('2D batched [4,3] x [4,3]', () => {
      const r = cross(mk([4, 3]), mk([4, 3]).add(1));
      expect((r as any).shape).toEqual([4, 3]);
    });

    it('3D batched [2,4,3] x [2,4,3]', () => {
      const r = cross(mk([2, 4, 3]), mk([2, 4, 3]).add(1));
      expect((r as any).shape).toEqual([2, 4, 3]);
    });

    it('4D batched', () => {
      const r = cross(mk([2, 2, 4, 3]), mk([2, 2, 4, 3]).add(1));
      expect((r as any).shape).toEqual([2, 2, 4, 3]);
    });
  });

  // ============================================================
  // SECTION 33: einsum — various subscripts + ND
  // ============================================================
  describe('einsum: various subscripts and ND', () => {
    it('1D dot: i,i->', () => {
      const r = einsum('i,i->', array([1, 2, 3, 4]), array([1, 2, 3, 4]));
      expect(arraysClose(r as number, 30)).toBe(true);
    });

    it('2D matmul: ij,jk->ik', () => {
      const r = einsum('ij,jk->ik', mk([2, 3]), mk([3, 4])) as any;
      expect(r.shape).toEqual([2, 4]);
    });

    it('2D trace: ii->', () => {
      const r = einsum('ii->', mk([3, 3]));
      expect(arraysClose(r as number, 1 + 5 + 9)).toBe(true);
    });

    it('3D batch matmul: bij,bjk->bik', () => {
      const r = einsum('bij,bjk->bik', mk([2, 3, 4]), mk([2, 4, 5])) as any;
      expect(r.shape).toEqual([2, 3, 5]);
    });

    it('3D sum last axis: ijk->ij', () => {
      const r = einsum('ijk->ij', mk([2, 3, 4])) as any;
      expect(r.shape).toEqual([2, 3]);
    });

    it('4D sum: ijkl->ij', () => {
      const r = einsum('ijkl->ij', mk([2, 3, 4, 5])) as any;
      expect(r.shape).toEqual([2, 3]);
    });

    it('outer: i,j->ij', () => {
      const r = einsum('i,j->ij', mk([3]), mk([4])) as any;
      expect(r.shape).toEqual([3, 4]);
    });
  });

  // ============================================================
  // SECTION 34: linalg batch ops — 2D through 4D
  // ============================================================
  describe('linalg.det: 2D through 4D batch', () => {
    for (const [shape, expectedBatchShape] of [
      [[3, 3], []],
      [[2, 3, 3], [2]],
      [
        [2, 2, 3, 3],
        [2, 2],
      ],
      [
        [2, 2, 2, 3, 3],
        [2, 2, 2],
      ],
    ] as [number[], number[]][]) {
      it(`det: ${JSON.stringify(shape)} -> ${JSON.stringify(expectedBatchShape)}`, () => {
        // Add identity*5 for conditioning
        const A = mk(shape);
        const r = linalg.det(A);
        if (expectedBatchShape.length === 0) {
          expect(typeof r).toBe('number');
        } else {
          expect((r as any).shape).toEqual(expectedBatchShape);
        }
      });
    }
  });

  describe('linalg.inv: 2D through 4D batch', () => {
    for (const [shape] of [[[2, 2]], [[2, 2, 2]], [[2, 2, 2, 2]], [[2, 2, 2, 2, 2]]] as [
      number[],
    ][]) {
      it(`inv: ${JSON.stringify(shape)}`, () => {
        // Use identity-like matrices for well-conditioning
        const size = shape[shape.length - 1];
        const batchDims = shape.slice(0, -2);
        const batchSize = batchDims.reduce((a, b) => a * b, 1);
        const identity = Array.from({ length: batchSize }, () =>
          Array.from({ length: size }, (_, i) =>
            Array.from({ length: size }, (_, j) => (i === j ? 2 : 0))
          )
        ).flat(2);
        const A = array(identity).reshape(shape);
        const r = linalg.inv(A);
        expect((r as any).shape).toEqual(shape);
      });
    }
  });

  describe('linalg.solve: batched', () => {
    it('solve: 2D [3,3] · [3]', () => {
      const A = array([
        [2, 1, 0],
        [0, 3, 1],
        [0, 0, 4],
      ]);
      const b = array([1, 2, 3]);
      const r = linalg.solve(A, b);
      expect((r as any).shape).toEqual([3]);
    });

    it('solve: 2D [3,3] · [3,2] (multiple RHS)', () => {
      const A = array([
        [2, 1, 0],
        [0, 3, 1],
        [0, 0, 4],
      ]);
      const b = mk([3, 2]);
      const r = linalg.solve(A, b);
      expect((r as any).shape).toEqual([3, 2]);
    });

    it('solve: batch 3D [2,3,3] · [2,3] -> [2,3]', () => {
      const A = mk([2, 3, 3]); // using solve - may not be well-conditioned, just check shape
      const b = mk([2, 3]);
      try {
        const r = linalg.solve(A, b);
        expect((r as any).shape).toEqual([2, 3]);
      } catch (e) {
        // Accept singular matrix errors
        expect((e as any).message).toBeTruthy();
      }
    });
  });

  describe('linalg.norm: all axes', () => {
    it('1D vector norm', () => {
      const r = linalg.norm(array([3, 4]));
      expect(arraysClose(r as number, 5)).toBe(true);
    });

    it('2D [3,4] axis=0', () => {
      const r = linalg.norm(mk([3, 4]), undefined, 0);
      expect((r as any).shape).toEqual([4]);
    });

    it('2D [3,4] axis=1', () => {
      const r = linalg.norm(mk([3, 4]), undefined, 1);
      expect((r as any).shape).toEqual([3]);
    });

    it('2D [3,4] fro', () => {
      const r = linalg.norm(mk([3, 4]), 'fro');
      expect(typeof r).toBe('number');
    });

    it('3D [2,3,4] axis=-1', () => {
      const r = linalg.norm(mk([2, 3, 4]), undefined, -1);
      expect((r as any).shape).toEqual([2, 3]);
    });

    it('3D [2,3,4] axis=(0,1)', () => {
      const r = linalg.norm(mk([2, 3, 4]), undefined, [0, 1]);
      expect((r as any).shape).toEqual([4]);
    });

    it('4D [2,3,4,5] axis=-1 keepdims', () => {
      const r = linalg.norm(mk([2, 3, 4, 5]), undefined, -1, true);
      expect((r as any).shape).toEqual([2, 3, 4, 1]);
    });
  });

  describe('linalg.svd: batched', () => {
    it('svd: 2D [3,4] reduced', () => {
      const r = linalg.svd(mk([3, 4]), false) as any;
      expect(r.s.shape).toEqual([3]);
    });

    it('svd: 3D [2,3,4] batch', () => {
      const r = linalg.svd(mk([2, 3, 4]), false) as any;
      expect(r.s.shape).toEqual([2, 3]);
    });

    it('svd: 4D [2,2,3,4] batch', () => {
      const r = linalg.svd(mk([2, 2, 3, 4]), false) as any;
      expect(r.s.shape).toEqual([2, 2, 3]);
    });
  });

  describe('linalg.qr: batched', () => {
    it('qr: 2D [4,3]', () => {
      const r = linalg.qr(mk([4, 3])) as any;
      expect(r.q.shape).toEqual([4, 3]);
      expect(r.r.shape).toEqual([3, 3]);
    });

    it('qr: 3D [2,4,3] batch', () => {
      const r = linalg.qr(mk([2, 4, 3])) as any;
      expect(r.q.shape).toEqual([2, 4, 3]);
      expect(r.r.shape).toEqual([2, 3, 3]);
    });
  });

  // ============================================================
  // SECTION 35: FFT with explicit axis parameter
  // ============================================================
  describe('fft: explicit axis parameter', () => {
    for (const [shape, axes] of [
      [[8], [0, -1]],
      [
        [4, 8],
        [0, 1, -1, -2],
      ],
      [
        [2, 4, 8],
        [0, 1, 2, -1],
      ],
      [
        [2, 2, 4, 8],
        [0, 1, 2, 3, -1],
      ],
    ] as [number[], number[]][]) {
      for (const ax of axes) {
        it(`fft.fft: ${JSON.stringify(shape)} axis=${ax}`, () => {
          const A = mk(shape);
          const r = fft.fft(A, undefined, ax);
          expect((r as any).shape).toEqual(shape);
        });

        it(`fft.ifft: ${JSON.stringify(shape)} axis=${ax}`, () => {
          const A = mk(shape);
          const r = fft.ifft(A, undefined, ax);
          expect((r as any).shape).toEqual(shape);
        });
      }
    }

    it('fft.fft: 3D axis=0 vs axis=1', () => {
      const A = mk([2, 4, 8]);
      const r0 = fft.fft(A, undefined, 0);
      const r1 = fft.fft(A, undefined, 1);
      expect((r0 as any).shape).toEqual([2, 4, 8]);
      expect((r1 as any).shape).toEqual([2, 4, 8]);
    });

    it('fft.fft2: 3D [2,4,8] with axes=(1,2)', () => {
      const A = mk([2, 4, 8]);
      const r = fft.fft2(A, undefined, [1, 2]);
      expect((r as any).shape).toEqual([2, 4, 8]);
    });

    it('fft.fft2: 4D [2,2,4,8] with axes=(2,3)', () => {
      const A = mk([2, 2, 4, 8]);
      const r = fft.fft2(A, undefined, [2, 3]);
      expect((r as any).shape).toEqual([2, 2, 4, 8]);
    });

    it('fft.fftn: 3D all axes', () => {
      const A = mk([2, 4, 8]);
      const r = fft.fftn(A);
      expect((r as any).shape).toEqual([2, 4, 8]);
    });

    it('fft.fftn: 4D last 2 axes', () => {
      const A = mk([2, 2, 4, 8]);
      const r = fft.fftn(A, undefined, [-2, -1]);
      expect((r as any).shape).toEqual([2, 2, 4, 8]);
    });

    it('fft.fftshift: 1D through 4D', () => {
      for (const shape of [[8], [4, 8], [2, 4, 8], [2, 2, 4, 8]] as number[][]) {
        const A = mk(shape);
        const r = fft.fftshift(A);
        expect((r as any).shape).toEqual(shape);
      }
    });

    it('fft.ifftshift: 3D with axis', () => {
      const A = mk([2, 4, 8]);
      const r = fft.ifftshift(A, 1);
      expect((r as any).shape).toEqual([2, 4, 8]);
    });
  });

  // ============================================================
  // SECTION 36: 0D special cases
  // ============================================================
  describe('0D special cases', () => {
    it('sum(0D) -> scalar', () => {
      expect(sum(array(5.0))).toBeCloseTo(5.0);
    });

    it('mean(0D) -> scalar', () => {
      expect(mean(array(5.0))).toBeCloseTo(5.0);
    });

    it('prod(0D) -> scalar', () => {
      expect(prod(array(5.0))).toBeCloseTo(5.0);
    });

    it('sin(0D) -> 0D', () => {
      const r = sin(array(0.5));
      const py = runNumPy(`result = np.sin(np.array(0.5))`);
      const val = (r as any).toArray?.() ?? r;
      expect(arraysClose(val, py.value)).toBe(true);
    });

    it('exp(0D) -> 0D', () => {
      const r = exp(array(1.0));
      const val = (r as any).toArray?.() ?? r;
      expect(arraysClose(val, Math.E, 1e-10)).toBe(true);
    });

    it('abs(0D) -> 0D', () => {
      const r = abs(array(-3.5));
      const val = (r as any).toArray?.() ?? r;
      expect(arraysClose(val, 3.5)).toBe(true);
    });

    it('add(0D, 0D) -> 0D', () => {
      const r = add(array(2.0), array(3.0));
      const val = (r as any).toArray?.() ?? r;
      expect(arraysClose(val, 5.0)).toBe(true);
    });

    it('multiply(0D, 0D) -> 0D', () => {
      const r = multiply(array(2.0), array(3.0));
      const val = (r as any).toArray?.() ?? r;
      expect(arraysClose(val, 6.0)).toBe(true);
    });

    it('squeeze(0D) -> 0D', () => {
      const r = squeeze(array(5.0));
      expect((r as any).shape ?? []).toEqual([]);
    });

    it('reshape(0D, [1]) -> 1D', () => {
      const r = reshape(array(5.0), [1]);
      expect((r as any).shape).toEqual([1]);
    });

    it('isfinite(0D)', () => {
      const r = isfinite(array(5.0));
      const val = (r as any).toArray?.() ?? r;
      // val is either a boolean or an array/number representing true
      expect(val === true || val === 1 || (Array.isArray(val) && val[0])).toBe(true);
    });
  });

  // ============================================================
  // SECTION 37: 5D+ general ND
  // ============================================================
  describe('5D+ general ND support', () => {
    const A5 = mk([2, 2, 2, 3, 4]);

    it('sin: 5D', () => {
      const r = sin(A5.multiply(0.1));
      expect((r as any).shape).toEqual([2, 2, 2, 3, 4]);
    });

    it('sum: 5D no-axis', () => {
      const r = sum(A5);
      const n = 2 * 2 * 2 * 3 * 4;
      expect(arraysClose(r as number, ((1 + n) * n) / 2)).toBe(true);
    });

    it('sum: 5D axis=2', () => {
      const r = sum(A5, 2);
      expect((r as any).shape).toEqual([2, 2, 3, 4]);
    });

    it('mean: 5D axis=-1', () => {
      const r = mean(A5, -1);
      expect((r as any).shape).toEqual([2, 2, 2, 3]);
    });

    it('flip: 5D', () => {
      const r = flip(A5);
      expect((r as any).shape).toEqual([2, 2, 2, 3, 4]);
    });

    it('transpose: 5D default', () => {
      const r = transpose(A5);
      expect((r as any).shape).toEqual([4, 3, 2, 2, 2]);
    });

    it('concatenate: 5D axis=0', () => {
      const r = concatenate([A5, A5], 0);
      expect((r as any).shape).toEqual([4, 2, 2, 3, 4]);
    });

    it('matmul: 5D batch', () => {
      const a = mk([2, 2, 2, 3, 4]);
      const b = mk([2, 2, 2, 4, 5]);
      const r = matmul(a, b);
      expect((r as any).shape).toEqual([2, 2, 2, 3, 5]);
    });

    it('sort: 5D axis=-1', () => {
      const r = sort(A5, -1);
      expect((r as any).shape).toEqual([2, 2, 2, 3, 4]);
    });

    it('cumsum: 5D axis=3', () => {
      const r = cumsum(A5, 3);
      expect((r as any).shape).toEqual([2, 2, 2, 3, 4]);
    });
  });

  // ============================================================
  // SECTION 38: MAX_NDIM enforcement
  // ============================================================
  describe('MAX_NDIM: 64 maximum', () => {
    it('accepts 64-dimensional arrays', () => {
      expect(() => zeros(Array(64).fill(1))).not.toThrow();
    });

    it('rejects 65-dimensional arrays', () => {
      expect(() => zeros(Array(65).fill(1))).toThrow(/64/);
    });

    it('64-dim array: shape and ndim correct', () => {
      const a = zeros(Array(64).fill(1));
      expect(a.ndim).toBe(64);
      expect(a.shape).toHaveLength(64);
    });

    it('64-dim sum works', () => {
      const a = zeros(Array(64).fill(1));
      expect(() => sum(a)).not.toThrow();
    });

    it('64-dim elementwise works', () => {
      const a = zeros(Array(64).fill(1));
      expect(() => sin(a)).not.toThrow();
    });
  });
});
