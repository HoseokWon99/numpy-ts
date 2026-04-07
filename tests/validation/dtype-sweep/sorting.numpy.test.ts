/**
 * DType Sweep: Sorting & searching functions.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import * as np from '../../../src';
import { SETS, runNumPy, arraysClose, checkNumPyAvailable, npDtype } from './_helpers';

const { array } = np;
const REAL = SETS.REAL;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');
});

describe('DType Sweep: Sorting', () => {
  for (const dtype of REAL) {
    it(`sort ${dtype}`, () => {
      const a = array([5, 2, 8, 1, 9, 3], dtype);
      const jsResult = np.sort(a);
      const pyResult = runNumPy(`
a = np.array([5, 2, 8, 1, 9, 3], dtype=${npDtype(dtype)})
result = np.sort(a).astype(np.float64)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`argsort ${dtype}`, () => {
      const a = array([5, 2, 8, 1, 9, 3], dtype);
      const jsResult = np.argsort(a);
      const pyResult = runNumPy(`
a = np.array([5, 2, 8, 1, 9, 3], dtype=${npDtype(dtype)})
result = np.argsort(a)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`searchsorted ${dtype}`, () => {
      const a = array([1, 3, 5, 7, 9], dtype);
      const jsResult = np.searchsorted(a, array([2, 4, 6], dtype));
      const pyResult = runNumPy(`
a = np.array([1, 3, 5, 7, 9], dtype=${npDtype(dtype)})
result = np.searchsorted(a, np.array([2, 4, 6], dtype=${npDtype(dtype)}))
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  }
});
