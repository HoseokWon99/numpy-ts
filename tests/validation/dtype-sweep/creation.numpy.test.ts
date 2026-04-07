/**
 * DType Sweep: Creation functions.
 * Validates that arrays are created with correct dtype and shape.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import * as np from '../../../src';
import { SETS, checkNumPyAvailable, isComplex } from './_helpers';

const { array } = np;
const ALL = SETS.ALL;
const REAL = SETS.REAL;
const FLOAT = SETS.FLOAT;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');
});

describe('DType Sweep: Creation', () => {
  for (const dtype of ALL) {
    it(`array ${dtype}`, () => {
      const a = array(isComplex(dtype) ? [1, 2, 3] : dtype === 'bool' ? [1, 0, 1] : [1, 2, 3], dtype);
      expect(a.dtype).toBe(dtype);
      expect(a.shape).toEqual([3]);
    });

    it(`zeros ${dtype}`, () => {
      expect(np.zeros([3], dtype).dtype).toBe(dtype);
    });

    it(`ones ${dtype}`, () => {
      expect(np.ones([3], dtype).dtype).toBe(dtype);
    });

    it(`full ${dtype}`, () => {
      expect(np.full([3], dtype === 'bool' ? 1 : 5, dtype).dtype).toBe(dtype);
    });

    it(`empty ${dtype}`, () => {
      expect(np.empty([3], dtype).dtype).toBe(dtype);
    });

    it(`eye ${dtype}`, () => {
      expect(np.eye(3, undefined, undefined, dtype).dtype).toBe(dtype);
    });

    it(`asarray ${dtype}`, () => {
      const a = array(dtype === 'bool' ? [1, 0] : isComplex(dtype) ? [1, 2] : [1, 2], dtype);
      expect(np.asarray(a).dtype).toBe(dtype);
    });

    it(`ascontiguousarray ${dtype}`, () => {
      const a = array(dtype === 'bool' ? [1, 0] : isComplex(dtype) ? [1, 2] : [1, 2], dtype);
      expect(np.ascontiguousarray(a).dtype).toBe(dtype);
    });
  }

  for (const dtype of REAL) {
    it(`arange ${dtype}`, () => {
      expect(np.arange(0, 5, 1, dtype).dtype).toBe(dtype);
    });
  }

  for (const dtype of FLOAT) {
    it(`linspace ${dtype}`, () => {
      expect(np.linspace(0, 1, 5, dtype).dtype).toBe(dtype);
    });

    it(`logspace ${dtype}`, () => {
      expect(np.logspace(0, 2, 5, undefined, dtype).dtype).toBe(dtype);
    });
  }
});
