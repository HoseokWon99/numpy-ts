/**
 * DType Sweep: Unary element-wise math functions.
 * Tests each function across its full valid dtype set, validated against NumPy.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import * as np from '../../../src';
import { SETS, runNumPy, arraysClose, checkNumPyAvailable, npDtype, isInt } from './_helpers';

const { array } = np;
const FLOAT = SETS.FLOAT;
const REAL = SETS.REAL;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');
});

describe('DType Sweep: Unary math', () => {
  const unaryOps: { name: string; fn: (a: any) => any; dtypes: readonly string[] }[] = [
    { name: 'absolute', fn: np.absolute, dtypes: REAL },
    { name: 'negative', fn: np.negative, dtypes: REAL },
    { name: 'positive', fn: np.positive, dtypes: REAL },
    { name: 'sign', fn: np.sign, dtypes: REAL },
    { name: 'square', fn: np.square, dtypes: REAL },
    { name: 'sqrt', fn: np.sqrt, dtypes: FLOAT },
    { name: 'cbrt', fn: np.cbrt, dtypes: REAL },
    { name: 'reciprocal', fn: np.reciprocal, dtypes: FLOAT },
    { name: 'exp', fn: np.exp, dtypes: REAL },
    { name: 'exp2', fn: np.exp2, dtypes: REAL },
    { name: 'expm1', fn: np.expm1, dtypes: REAL },
    { name: 'log', fn: np.log, dtypes: FLOAT },
    { name: 'log2', fn: np.log2, dtypes: FLOAT },
    { name: 'log10', fn: np.log10, dtypes: FLOAT },
    { name: 'log1p', fn: np.log1p, dtypes: FLOAT },
    { name: 'sin', fn: np.sin, dtypes: REAL },
    { name: 'cos', fn: np.cos, dtypes: REAL },
    { name: 'tan', fn: np.tan, dtypes: REAL },
    { name: 'arcsin', fn: np.arcsin, dtypes: REAL },
    { name: 'arccos', fn: np.arccos, dtypes: REAL },
    { name: 'arctan', fn: np.arctan, dtypes: REAL },
    { name: 'sinh', fn: np.sinh, dtypes: REAL },
    { name: 'cosh', fn: np.cosh, dtypes: REAL },
    { name: 'tanh', fn: np.tanh, dtypes: REAL },
    { name: 'arcsinh', fn: np.arcsinh, dtypes: REAL },
    { name: 'arccosh', fn: np.arccosh, dtypes: FLOAT },
    { name: 'arctanh', fn: np.arctanh, dtypes: REAL },
    { name: 'ceil', fn: np.ceil, dtypes: REAL },
    { name: 'floor', fn: np.floor, dtypes: REAL },
    { name: 'rint', fn: np.rint, dtypes: REAL },
    { name: 'trunc', fn: np.trunc, dtypes: REAL },
    { name: 'fix', fn: np.fix, dtypes: REAL },
    { name: 'degrees', fn: np.degrees, dtypes: REAL },
    { name: 'radians', fn: np.radians, dtypes: REAL },
    { name: 'sinc', fn: np.sinc, dtypes: REAL },
    { name: 'fabs', fn: np.fabs, dtypes: REAL },
    { name: 'signbit', fn: np.signbit, dtypes: REAL },
  ];

  for (const { name, fn, dtypes } of unaryOps) {
    describe(name, () => {
      for (const dtype of dtypes) {
        it(`${dtype}`, () => {
          const needsDomain = ['arcsin', 'arccos', 'arctanh'].includes(name);
          const needsPositive = ['arccosh', 'log', 'log2', 'log10', 'log1p', 'sqrt'].includes(name);
          const data = needsDomain
            ? (isInt(dtype) ? [0, 1, 0, 1] : [0.1, 0.5, 0.9, 0.3])
            : needsPositive
              ? [1, 2, 3, 4]
              : [1, 2, 3, 4];
          const a = array(data, dtype);
          const jsResult = fn(a);
          const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
result = np.${name}(a).astype(np.float64)
          `);
          const rtol = dtype === 'float32' ? 1e-2 : 1e-3;
          const atol = dtype === 'float32' ? 1e-5 : 1e-8;
          expect(arraysClose(jsResult.toArray(), pyResult.value, rtol, atol)).toBe(true);
        });
      }
    });
  }
});
