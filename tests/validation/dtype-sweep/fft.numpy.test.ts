/**
 * DType Sweep: FFT functions.
 * Tests FFT operations across ALL dtypes, validated against NumPy.
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

describe('DType Sweep: FFT', () => {
  for (const dtype of ALL_DTYPES) {
    const data = dtype === 'bool' ? [1, 0, 1, 0] : [1, 2, 3, 4];
    const data2d =
      dtype === 'bool'
        ? [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
          ]
        : [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
          ];
    const tol = dtype === 'float32' || dtype === 'complex64' ? 1e-2 : 1e-4;

    it(`fft.fft ${dtype}`, () => {
      const jsResult = np.fft.fft(array(data, dtype));
      const pyResult = runNumPy(`
result = np.fft.fft(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})).astype(np.complex128)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value, tol)).toBe(true);
    });

    it(`fft.ifft ${dtype}`, () => {
      // ifft(fft(x)) round-trip
      const fftResult = np.fft.fft(array(data, dtype));
      const jsResult = np.fft.ifft(fftResult);
      const pyResult = runNumPy(`
result = np.fft.ifft(np.fft.fft(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)}))).astype(np.complex128)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value, tol)).toBe(true);
    });

    it(`fft.rfft ${dtype}`, () => {
      const pyCode = `result = np.fft.rfft(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})).astype(np.complex128)`;
      // rfft: expects real input — complex dtypes rejected by NumPy
      if (isComplex(dtype)) {
        { const _r = expectBothReject('rfft expects real-valued input, not complex', () => np.fft.rfft(array(data, dtype)), pyCode); if (_r === 'both-reject') return; }
      }
      const jsResult = np.fft.rfft(array(data, dtype));
      const pyResult = runNumPy(pyCode);
      expect(arraysClose(jsResult.toArray(), pyResult.value, tol)).toBe(true);
    });

    it(`fft.irfft ${dtype}`, () => {
      const pyCode = `result = np.fft.irfft(np.fft.rfft(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})))`;
      // irfft round-trip through rfft — complex input to rfft rejected
      if (isComplex(dtype)) {
        { const _r = expectBothReject('rfft(complex) not supported, so irfft round-trip fails', () => { const r = np.fft.rfft(array(data, dtype)); np.fft.irfft(r); }, pyCode); if (_r === 'both-reject') return; }
      }
      const rfftResult = np.fft.rfft(array(data, dtype));
      const jsResult = np.fft.irfft(rfftResult);
      const pyResult = runNumPy(pyCode);
      expect(arraysClose(jsResult.toArray(), pyResult.value, tol)).toBe(true);
    });

    it(`fft.fft2 ${dtype}`, () => {
      const jsResult = np.fft.fft2(array(data2d, dtype));
      const pyResult = runNumPy(`
result = np.fft.fft2(np.array(${JSON.stringify(data2d)}, dtype=${npDtype(dtype)})).astype(np.complex128)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value, tol)).toBe(true);
    });

    it(`fft.ifft2 ${dtype}`, () => {
      const fft2Result = np.fft.fft2(array(data2d, dtype));
      const jsResult = np.fft.ifft2(fft2Result);
      const pyResult = runNumPy(`
result = np.fft.ifft2(np.fft.fft2(np.array(${JSON.stringify(data2d)}, dtype=${npDtype(dtype)}))).astype(np.complex128)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value, tol)).toBe(true);
    });

    it(`fft.rfft2 ${dtype}`, () => {
      const pyCode = `result = np.fft.rfft2(np.array(${JSON.stringify(data2d)}, dtype=${npDtype(dtype)})).astype(np.complex128)`;
      // rfft2: expects real input — complex rejected
      if (isComplex(dtype)) {
        { const _r = expectBothReject('rfft2 expects real-valued input', () => np.fft.rfft2(array(data2d, dtype)), pyCode); if (_r === 'both-reject') return; }
      }
      const jsResult = np.fft.rfft2(array(data2d, dtype));
      const pyResult = runNumPy(pyCode);
      expect(arraysClose(jsResult.toArray(), pyResult.value, tol)).toBe(true);
    });

    it(`fft.irfft2 ${dtype}`, () => {
      const pyCode = `result = np.fft.irfft2(np.fft.rfft2(np.array(${JSON.stringify(data2d)}, dtype=${npDtype(dtype)})))`;
      if (isComplex(dtype)) {
        { const _r = expectBothReject('rfft2(complex) not supported', () => { const r = np.fft.rfft2(array(data2d, dtype)); np.fft.irfft2(r); }, pyCode); if (_r === 'both-reject') return; }
      }
      const rfft2Result = np.fft.rfft2(array(data2d, dtype));
      const jsResult = np.fft.irfft2(rfft2Result);
      const pyResult = runNumPy(pyCode);
      expect(arraysClose(jsResult.toArray(), pyResult.value, tol)).toBe(true);
    });

    it(`fft.fftn ${dtype}`, () => {
      const jsResult = np.fft.fftn(array(data2d, dtype));
      const pyResult = runNumPy(`
result = np.fft.fftn(np.array(${JSON.stringify(data2d)}, dtype=${npDtype(dtype)})).astype(np.complex128)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value, tol)).toBe(true);
    });

    it(`fft.ifftn ${dtype}`, () => {
      const fftnResult = np.fft.fftn(array(data2d, dtype));
      const jsResult = np.fft.ifftn(fftnResult);
      const pyResult = runNumPy(`
result = np.fft.ifftn(np.fft.fftn(np.array(${JSON.stringify(data2d)}, dtype=${npDtype(dtype)}))).astype(np.complex128)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value, tol)).toBe(true);
    });

    it(`fft.rfftn ${dtype}`, () => {
      const pyCode = `result = np.fft.rfftn(np.array(${JSON.stringify(data2d)}, dtype=${npDtype(dtype)})).astype(np.complex128)`;
      // rfftn: expects real input — complex rejected
      if (isComplex(dtype)) {
        { const _r = expectBothReject('rfftn expects real-valued input', () => np.fft.rfftn(array(data2d, dtype)), pyCode); if (_r === 'both-reject') return; }
      }
      const jsResult = np.fft.rfftn(array(data2d, dtype));
      const pyResult = runNumPy(pyCode);
      expect(arraysClose(jsResult.toArray(), pyResult.value, tol)).toBe(true);
    });

    it(`fft.irfftn ${dtype}`, () => {
      const pyCode = `result = np.fft.irfftn(np.fft.rfftn(np.array(${JSON.stringify(data2d)}, dtype=${npDtype(dtype)})))`;
      if (isComplex(dtype)) {
        { const _r = expectBothReject('rfftn(complex) not supported', () => { const r = np.fft.rfftn(array(data2d, dtype)); np.fft.irfftn(r); }, pyCode); if (_r === 'both-reject') return; }
      }
      const rfftnResult = np.fft.rfftn(array(data2d, dtype));
      const jsResult = np.fft.irfftn(rfftnResult);
      const pyResult = runNumPy(pyCode);
      expect(arraysClose(jsResult.toArray(), pyResult.value, tol)).toBe(true);
    });

    it(`fft.hfft ${dtype}`, () => {
      const jsResult = np.fft.hfft(array(data, dtype));
      const pyResult = runNumPy(`
result = np.fft.hfft(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)}))
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value, tol)).toBe(true);
    });

    it(`fft.ihfft ${dtype}`, () => {
      const pyCode = `result = np.fft.ihfft(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})).astype(np.complex128)`;
      // ihfft: expects real (Hermitian-symmetric) input — complex rejected by NumPy
      if (isComplex(dtype)) {
        { const _r = expectBothReject('ihfft expects real-valued input', () => np.fft.ihfft(array(data, dtype)), pyCode); if (_r === 'both-reject') return; }
      }
      const jsResult = np.fft.ihfft(array(data, dtype));
      const pyResult = runNumPy(pyCode);
      expect(arraysClose(jsResult.toArray(), pyResult.value, tol)).toBe(true);
    });

    it(`fft.fftfreq ${dtype}`, () => {
      const jsResult = np.fft.fftfreq(4);
      const pyResult = runNumPy(`result = np.fft.fftfreq(4)`);
      expect(arraysClose(jsResult.toArray(), pyResult.value, 1e-10)).toBe(true);
    });

    it(`fft.rfftfreq ${dtype}`, () => {
      const jsResult = np.fft.rfftfreq(4);
      const pyResult = runNumPy(`result = np.fft.rfftfreq(4)`);
      expect(arraysClose(jsResult.toArray(), pyResult.value, 1e-10)).toBe(true);
    });

    it(`fft.fftshift ${dtype}`, () => {
      const jsResult = np.fft.fftshift(array(data, dtype));
      const pyResult = runNumPy(
        `result = np.fft.fftshift(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})).astype(np.float64)`
      );
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`fft.ifftshift ${dtype}`, () => {
      const jsResult = np.fft.ifftshift(array(data, dtype));
      const pyResult = runNumPy(
        `result = np.fft.ifftshift(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})).astype(np.float64)`
      );
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });
  }
});
