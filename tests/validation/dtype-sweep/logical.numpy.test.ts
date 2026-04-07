/**
 * DType Sweep: Logical and bitwise operations, validated against NumPy.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import * as np from '../../../src';
import {
  ALL_DTYPES,
  runNumPy,
  arraysClose,
  checkNumPyAvailable,
  npDtype,
  isComplex,
  expectBothReject,
} from './_helpers';

const { array } = np;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');
});

describe('DType Sweep: Logical', () => {
  for (const dtype of ALL_DTYPES) {
    const data1 = dtype === 'bool' ? [1, 1, 0, 0] : isComplex(dtype) ? [1, 0, 1, 0] : [1, 1, 0, 0];
    const data2 = dtype === 'bool' ? [1, 0, 1, 0] : isComplex(dtype) ? [1, 1, 0, 0] : [1, 0, 1, 0];

    it(`logical_and ${dtype}`, () => {
      const jsResult = np.logical_and(array(data1, dtype), array(data2, dtype));
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(data2)}, dtype=${npDtype(dtype)})
result = np.logical_and(a, b).astype(np.float64)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`logical_or ${dtype}`, () => {
      const jsResult = np.logical_or(array(data1, dtype), array(data2, dtype));
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(data2)}, dtype=${npDtype(dtype)})
result = np.logical_or(a, b).astype(np.float64)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`logical_not ${dtype}`, () => {
      const jsResult = np.logical_not(array(data1, dtype));
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)})
result = np.logical_not(a).astype(np.float64)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`logical_xor ${dtype}`, () => {
      const jsResult = np.logical_xor(array(data1, dtype), array(data2, dtype));
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(data2)}, dtype=${npDtype(dtype)})
result = np.logical_xor(a, b).astype(np.float64)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`isnan ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const jsResult = np.isnan(array(data, dtype));
      const pyResult = runNumPy(`
result = np.isnan(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})).astype(np.float64)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`isinf ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const jsResult = np.isinf(array(data, dtype));
      const pyResult = runNumPy(`
result = np.isinf(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})).astype(np.float64)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`isfinite ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const jsResult = np.isfinite(array(data, dtype));
      const pyResult = runNumPy(`
result = np.isfinite(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})).astype(np.float64)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`isneginf ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const pyCode = `result = np.isneginf(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})).astype(np.float64)`;
      // isneginf: complex dtypes rejected by NumPy ("not supported for the input types")
      if (isComplex(dtype)) {
        { const _r = expectBothReject('isneginf is not defined for complex numbers', () => np.isneginf(array(data, dtype)), pyCode); if (_r === 'both-reject') return; }
      }
      const jsResult = np.isneginf(array(data, dtype));
      const pyResult = runNumPy(pyCode);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`isposinf ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const pyCode = `result = np.isposinf(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})).astype(np.float64)`;
      // isposinf: complex dtypes rejected by NumPy ("not supported for the input types")
      if (isComplex(dtype)) {
        { const _r = expectBothReject('isposinf is not defined for complex numbers', () => np.isposinf(array(data, dtype)), pyCode); if (_r === 'both-reject') return; }
      }
      const jsResult = np.isposinf(array(data, dtype));
      const pyResult = runNumPy(pyCode);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  }
});

describe('DType Sweep: Bitwise', () => {
  // Bitwise operations are only defined for integer + bool dtypes.
  // Both JS and NumPy reject float and complex inputs with TypeError.
  const BITWISE_UNSUPPORTED = (d: string) => d.startsWith('float') || d.startsWith('complex');

  for (const dtype of ALL_DTYPES) {
    const data1 = dtype === 'bool' ? [1, 1, 0, 0] : [15, 7, 3, 1];
    const data2 = dtype === 'bool' ? [1, 0, 1, 0] : [9, 6, 5, 3];

    it(`bitwise_and ${dtype}`, () => {
      const pyCode = `result = np.bitwise_and(np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(data2)}, dtype=${npDtype(dtype)})).astype(np.float64)`;
      // float/complex: bitwise ops not defined — both JS and NumPy throw TypeError
      if (BITWISE_UNSUPPORTED(dtype)) {
        { const _r = expectBothReject('bitwise_and requires integer or bool dtype', () => np.bitwise_and(array(data1, dtype), array(data2, dtype)), pyCode); if (_r === 'both-reject') return; }
      }
      const jsResult = np.bitwise_and(array(data1, dtype), array(data2, dtype));
      const pyResult = runNumPy(pyCode);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`bitwise_or ${dtype}`, () => {
      const pyCode = `result = np.bitwise_or(np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(data2)}, dtype=${npDtype(dtype)})).astype(np.float64)`;
      if (BITWISE_UNSUPPORTED(dtype)) {
        { const _r = expectBothReject('bitwise_or requires integer or bool dtype', () => np.bitwise_or(array(data1, dtype), array(data2, dtype)), pyCode); if (_r === 'both-reject') return; }
      }
      const jsResult = np.bitwise_or(array(data1, dtype), array(data2, dtype));
      const pyResult = runNumPy(pyCode);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`bitwise_xor ${dtype}`, () => {
      const pyCode = `result = np.bitwise_xor(np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(data2)}, dtype=${npDtype(dtype)})).astype(np.float64)`;
      if (BITWISE_UNSUPPORTED(dtype)) {
        { const _r = expectBothReject('bitwise_xor requires integer or bool dtype', () => np.bitwise_xor(array(data1, dtype), array(data2, dtype)), pyCode); if (_r === 'both-reject') return; }
      }
      const jsResult = np.bitwise_xor(array(data1, dtype), array(data2, dtype));
      const pyResult = runNumPy(pyCode);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`bitwise_not ${dtype}`, () => {
      const pyCode = `result = np.bitwise_not(np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)})).astype(np.float64)`;
      // float/complex: bitwise_not not defined — both reject
      if (BITWISE_UNSUPPORTED(dtype)) {
        { const _r = expectBothReject('bitwise_not requires integer or bool dtype', () => np.bitwise_not(array(data1, dtype)), pyCode); if (_r === 'both-reject') return; }
      }
      const jsResult = np.bitwise_not(array(data1, dtype));
      const pyResult = runNumPy(pyCode);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`invert ${dtype}`, () => {
      const pyCode = `result = np.invert(np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)})).astype(np.float64)`;
      if (BITWISE_UNSUPPORTED(dtype)) {
        { const _r = expectBothReject('invert requires integer or bool dtype', () => np.invert(array(data1, dtype)), pyCode); if (_r === 'both-reject') return; }
      }
      const jsResult = np.invert(array(data1, dtype));
      const pyResult = runNumPy(pyCode);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`left_shift ${dtype}`, () => {
      const d1 = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const d2 = dtype === 'bool' ? [1, 1, 0] : [1, 1, 1];
      const pyCode = `result = np.left_shift(np.array(${JSON.stringify(d1)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(d2)}, dtype=${npDtype(dtype)})).astype(np.float64)`;
      // float/complex: shift ops not defined — both reject
      if (BITWISE_UNSUPPORTED(dtype)) {
        { const _r = expectBothReject('left_shift requires integer or bool dtype', () => np.left_shift(array(d1, dtype), array(d2, dtype)), pyCode); if (_r === 'both-reject') return; }
      }
      const jsResult = np.left_shift(array(d1, dtype), array(d2, dtype));
      const pyResult = runNumPy(pyCode);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`right_shift ${dtype}`, () => {
      const d1 = dtype === 'bool' ? [1, 0, 1] : [8, 16, 32];
      const d2 = dtype === 'bool' ? [1, 0, 0] : [1, 1, 1];
      const pyCode = `result = np.right_shift(np.array(${JSON.stringify(d1)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(d2)}, dtype=${npDtype(dtype)})).astype(np.float64)`;
      if (BITWISE_UNSUPPORTED(dtype)) {
        { const _r = expectBothReject('right_shift requires integer or bool dtype', () => np.right_shift(array(d1, dtype), array(d2, dtype)), pyCode); if (_r === 'both-reject') return; }
      }
      const jsResult = np.right_shift(array(d1, dtype), array(d2, dtype));
      const pyResult = runNumPy(pyCode);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`bitwise_count ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1] : [7, 15, 3];
      const pyCode = `result = np.bitwise_count(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})).astype(np.float64)`;
      // float/complex: bitwise_count not defined — both reject
      if (BITWISE_UNSUPPORTED(dtype)) {
        { const _r = expectBothReject('bitwise_count requires integer or bool dtype', () => np.bitwise_count(array(data, dtype)), pyCode); if (_r === 'both-reject') return; }
      }
      const jsResult = np.bitwise_count(array(data, dtype));
      const pyResult = runNumPy(pyCode);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  }
});

describe('DType Sweep: Packbits/Unpackbits', () => {
  // packbits/unpackbits: NumPy only accepts uint8. Both JS and NumPy reject other dtypes.
  for (const dtype of ALL_DTYPES) {
    it(`packbits ${dtype}`, () => {
      const data = [1, 0, 1, 0, 1, 0, 1, 0];
      const pyCode = `result = np.packbits(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})).astype(np.float64)`;
      if (dtype !== 'uint8') {
        // Non-uint8: both JS and NumPy should reject (packbits only accepts uint8)
        { const _r = expectBothReject('packbits only accepts uint8 arrays', () => np.packbits(array(data, dtype)), pyCode); if (_r === 'both-reject') return; }
      }
      const jsResult = np.packbits(array(data, dtype));
      const pyResult = runNumPy(pyCode);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`unpackbits ${dtype}`, () => {
      const data = [170]; // 10101010 in binary
      const pyCode = `result = np.unpackbits(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})).astype(np.float64)`;
      if (dtype !== 'uint8') {
        // Non-uint8: both JS and NumPy should reject (unpackbits only accepts uint8)
        { const _r = expectBothReject('unpackbits only accepts uint8 arrays', () => np.unpackbits(array(data, dtype)), pyCode); if (_r === 'both-reject') return; }
      }
      const jsResult = np.unpackbits(array(data, dtype));
      const pyResult = runNumPy(pyCode);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  }
});
