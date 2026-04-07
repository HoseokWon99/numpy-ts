/**
 * DType Sweep: Linear algebra functions.
 * Tests linalg namespace functions across float dtypes, validated against NumPy.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import * as np from '../../../src';
import { SETS, runNumPy, arraysClose, checkNumPyAvailable, npDtype } from './_helpers';

const { array } = np;
const FLOAT = SETS.FLOAT;
const REAL = SETS.REAL;
const NUMERIC = SETS.NUMERIC;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');
});

describe('DType Sweep: Linalg', () => {
  for (const dtype of FLOAT) {
    it(`linalg.norm ${dtype}`, () => {
      const jsResult = np.linalg.norm(array([3, 4], dtype));
      const pyResult = runNumPy(`result = float(np.linalg.norm(np.array([3, 4], dtype=${npDtype(dtype)})))`);
      expect(Number(jsResult)).toBeCloseTo(Number(pyResult.value), 4);
    });

    it(`linalg.det ${dtype}`, () => {
      const jsResult = np.linalg.det(array([[1, 2], [3, 4]], dtype));
      const pyResult = runNumPy(`result = float(np.linalg.det(np.array([[1,2],[3,4]], dtype=${npDtype(dtype)})))`);
      expect(Number(jsResult)).toBeCloseTo(Number(pyResult.value), 4);
    });

    it(`linalg.inv ${dtype}`, () => {
      const jsResult = np.linalg.inv(array([[1, 2], [3, 4]], dtype));
      const pyResult = runNumPy(`result = np.linalg.inv(np.array([[1,2],[3,4]], dtype=${npDtype(dtype)})).astype(np.float64)`);
      expect(arraysClose(jsResult.toArray(), pyResult.value, 1e-4)).toBe(true);
    });

    it(`linalg.cholesky ${dtype}`, () => {
      const jsResult = np.linalg.cholesky(array([[4, 2], [2, 5]], dtype));
      const pyResult = runNumPy(`result = np.linalg.cholesky(np.array([[4,2],[2,5]], dtype=${npDtype(dtype)})).astype(np.float64)`);
      expect(arraysClose(jsResult.toArray(), pyResult.value, 1e-4)).toBe(true);
    });

    it(`linalg.qr ${dtype}`, () => {
      const { q, r } = np.linalg.qr(array([[1, 2], [3, 4]], dtype)) as any;
      expect(q.shape).toEqual([2, 2]);
      expect(r.shape).toEqual([2, 2]);
    });

    it(`linalg.svd ${dtype}`, () => {
      const { u, s, vh } = np.linalg.svd(array([[1, 2], [3, 4]], dtype)) as any;
      const pyResult = runNumPy(`
u, s, vh = np.linalg.svd(np.array([[1,2],[3,4]], dtype=${npDtype(dtype)}))
result = s.astype(np.float64)
      `);
      expect(arraysClose(s.toArray(), pyResult.value, 1e-4)).toBe(true);
    });

    it(`linalg.eig ${dtype}`, { timeout: 30000 }, () => {
      const { w } = np.linalg.eig(array([[1, 2], [3, 4]], dtype)) as any;
      expect(w.shape).toEqual([2]);
    });

    it(`linalg.solve ${dtype}`, () => {
      const jsResult = np.linalg.solve(array([[3, 1], [1, 2]], dtype), array([9, 8], dtype));
      const pyResult = runNumPy(`result = np.linalg.solve(np.array([[3,1],[1,2]], dtype=${npDtype(dtype)}), np.array([9,8], dtype=${npDtype(dtype)})).astype(np.float64)`);
      expect(arraysClose(jsResult.toArray(), pyResult.value, 1e-4)).toBe(true);
    });

    it(`linalg.lstsq ${dtype}`, () => {
      const { x } = np.linalg.lstsq(array([[1, 1], [1, 2], [1, 3]], dtype), array([1, 2, 3], dtype)) as any;
      expect(x.shape).toEqual([2]);
    });

    it(`linalg.matrix_rank ${dtype}`, () => {
      const jsResult = np.linalg.matrix_rank(array([[1, 2], [3, 4]], dtype));
      expect(Number(jsResult)).toBe(2);
    });

    it(`linalg.pinv ${dtype}`, () => {
      expect(np.linalg.pinv(array([[1, 2], [3, 4]], dtype)).shape).toEqual([2, 2]);
    });

    it(`linalg.cond ${dtype}`, () => {
      expect(typeof np.linalg.cond(array([[1, 2], [3, 4]], dtype))).toBe('number');
    });

    it(`linalg.slogdet ${dtype}`, () => {
      const { sign, logabsdet } = np.linalg.slogdet(array([[1, 2], [3, 4]], dtype)) as any;
      expect(typeof sign).toBe('number');
      expect(typeof logabsdet).toBe('number');
    });
  }
});

describe('DType Sweep: Top-level linalg', () => {
  // Skip int64/uint64 — bigint results can't round-trip through NumPy oracle JSON
  const REAL_NO_BIGINT = REAL.filter(d => d !== 'int64' && d !== 'uint64');
  for (const dtype of REAL_NO_BIGINT) {
    it(`outer ${dtype}`, () => {
      const a = array([1, 2, 3], dtype);
      const r = np.outer(a, a);
      const pyResult = runNumPy(`
a = np.array([1, 2, 3], dtype=${npDtype(dtype)})
result = np.outer(a, a).astype(np.float64)
      `);
      expect(arraysClose(r.toArray(), pyResult.value, 1e-4)).toBe(true);
    });

    it(`cross ${dtype}`, () => {
      const a = array([1, 2, 3], dtype);
      const b = array([4, 5, 6], dtype);
      const r = np.cross(a, b);
      const pyResult = runNumPy(`
a = np.array([1, 2, 3], dtype=${npDtype(dtype)})
b = np.array([4, 5, 6], dtype=${npDtype(dtype)})
result = np.cross(a, b).astype(np.float64)
      `);
      expect(arraysClose(r.toArray(), pyResult.value, 1e-4)).toBe(true);
    });

    it(`kron ${dtype}`, () => {
      const a = array([1, 2], dtype);
      const b = array([3, 4], dtype);
      const r = np.kron(a, b);
      const pyResult = runNumPy(`
a = np.array([1, 2], dtype=${npDtype(dtype)})
b = np.array([3, 4], dtype=${npDtype(dtype)})
result = np.kron(a, b).astype(np.float64)
      `);
      expect(arraysClose(r.toArray(), pyResult.value, 1e-4)).toBe(true);
    });

    it(`trace ${dtype}`, () => {
      const a = array([[1, 2], [3, 4]], dtype);
      const jsResult = np.trace(a);
      const pyResult = runNumPy(`result = float(np.trace(np.array([[1,2],[3,4]], dtype=${npDtype(dtype)})))`);
      expect(Number(jsResult)).toBeCloseTo(Number(pyResult.value), 4);
    });

    it(`tensordot ${dtype}`, () => {
      const a = array([[1, 2], [3, 4]], dtype);
      const b = array([[5, 6], [7, 8]], dtype);
      const r = np.tensordot(a, b);
      const pyResult = runNumPy(`
a = np.array([[1,2],[3,4]], dtype=${npDtype(dtype)})
b = np.array([[5,6],[7,8]], dtype=${npDtype(dtype)})
result = float(np.tensordot(a, b))
      `);
      expect(Number(r)).toBeCloseTo(Number(pyResult.value), 4);
    });
  }
});

