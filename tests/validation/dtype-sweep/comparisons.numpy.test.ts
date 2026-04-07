/**
 * DType Sweep: Comparison functions.
 * Tests each function across all valid dtypes, validated against NumPy.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import * as np from '../../../src';
import { SETS, runNumPy, arraysClose, checkNumPyAvailable, npDtype, isComplex } from './_helpers';

const { array } = np;
const ALL = SETS.ALL;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');
});

describe('DType Sweep: Comparisons', () => {
  const compOps = ['greater', 'greater_equal', 'less', 'less_equal', 'equal', 'not_equal'];

  for (const name of compOps) {
    describe(name, () => {
      for (const dtype of ALL) {
        if (isComplex(dtype)) continue;
        it(`${dtype}`, () => {
          const data1 = dtype === 'bool' ? [1, 0, 1, 0] : [1, 2, 3, 4];
          const data2 = dtype === 'bool' ? [0, 1, 1, 0] : [4, 3, 2, 1];
          const a = array(data1, dtype);
          const b = array(data2, dtype);
          const jsResult = (np as any)[name](a, b);
          const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(data2)}, dtype=${npDtype(dtype)})
result = np.${name}(a, b).astype(np.float64)
          `);
          expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
        });
      }
    });
  }
});
