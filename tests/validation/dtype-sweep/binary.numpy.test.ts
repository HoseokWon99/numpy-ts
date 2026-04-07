/**
 * DType Sweep: Binary element-wise arithmetic functions.
 * Tests each function across its full valid dtype set, validated against NumPy.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import * as np from '../../../src';
import { SETS, runNumPy, arraysClose, checkNumPyAvailable, npDtype } from './_helpers';

const { array } = np;
const FLOAT = SETS.FLOAT;
const REAL = SETS.REAL;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');
});

describe('DType Sweep: Binary arithmetic', () => {
  const binaryOps: { name: string; fn: (a: any, b: any) => any; dtypes: readonly string[] }[] = [
    { name: 'add', fn: np.add, dtypes: REAL },
    { name: 'subtract', fn: np.subtract, dtypes: REAL },
    { name: 'multiply', fn: np.multiply, dtypes: REAL },
    { name: 'divide', fn: np.divide, dtypes: REAL },
    { name: 'power', fn: np.power, dtypes: REAL },
    { name: 'maximum', fn: np.maximum, dtypes: REAL },
    { name: 'minimum', fn: np.minimum, dtypes: REAL },
    { name: 'mod', fn: np.mod, dtypes: REAL },
    { name: 'floor_divide', fn: np.floor_divide, dtypes: REAL },
    { name: 'copysign', fn: np.copysign, dtypes: FLOAT },
    { name: 'hypot', fn: np.hypot, dtypes: REAL },
    { name: 'arctan2', fn: np.arctan2, dtypes: REAL },
    { name: 'logaddexp', fn: np.logaddexp, dtypes: REAL },
    { name: 'fmax', fn: np.fmax, dtypes: REAL },
    { name: 'fmin', fn: np.fmin, dtypes: REAL },
    { name: 'remainder', fn: np.remainder, dtypes: REAL },
    { name: 'float_power', fn: np.float_power, dtypes: FLOAT }, // int overflow issues
    { name: 'heaviside', fn: np.heaviside, dtypes: FLOAT },
    { name: 'gcd', fn: np.gcd, dtypes: SETS.INTEGER },
    { name: 'lcm', fn: np.lcm, dtypes: SETS.INTEGER },
  ];

  for (const { name, fn, dtypes } of binaryOps) {
    describe(name, () => {
      for (const dtype of dtypes) {
        it(`${dtype}`, () => {
          const a = array([6, 7, 8, 9], dtype);
          const b = array([1, 2, 3, 4], dtype);
          const jsResult = fn(a, b);
          const pyResult = runNumPy(`
a = np.array([6, 7, 8, 9], dtype=${npDtype(dtype)})
b = np.array([1, 2, 3, 4], dtype=${npDtype(dtype)})
result = np.${name}(a, b).astype(np.float64)
          `);
          expect(arraysClose(jsResult.toArray(), pyResult.value, 1e-3)).toBe(true);
        });
      }
    });
  }
});
