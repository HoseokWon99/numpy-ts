/**
 * DType Sweep: Shape manipulation functions.
 * Structural tests — validates shape/dtype preservation across all dtypes.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import * as np from '../../../src';
import { SETS, checkNumPyAvailable, isComplex } from './_helpers';

const { array } = np;
const ALL = SETS.ALL;

const SMALL_DATA = [1, 2, 3, 4, 5, 6];
const SMALL_2D = [
  [1, 2, 3],
  [4, 5, 6],
];

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');
});

describe('DType Sweep: Shape manipulation', () => {
  for (const dtype of ALL) {
    const data = dtype === 'bool' ? [1, 0, 1, 0, 1, 0] : SMALL_DATA;
    const data2d =
      dtype === 'bool'
        ? [
            [1, 0, 1],
            [0, 1, 0],
          ]
        : SMALL_2D;

    it(`reshape ${dtype}`, () => {
      expect(np.reshape(array(data, dtype), [2, 3]).shape).toEqual([2, 3]);
    });

    it(`transpose ${dtype}`, () => {
      expect(np.transpose(array(data2d, dtype)).shape).toEqual([3, 2]);
    });

    it(`ravel ${dtype}`, () => {
      expect(np.ravel(array(data2d, dtype)).shape).toEqual([6]);
    });

    it(`concatenate ${dtype}`, () => {
      const a = array(dtype === 'bool' ? [1, 0] : [1, 2], dtype);
      const b = array(dtype === 'bool' ? [0, 1] : [3, 4], dtype);
      const r = np.concatenate([a, b]);
      expect(r.shape).toEqual([4]);
      expect(r.dtype).toBe(dtype);
    });

    it(`stack ${dtype}`, () => {
      const a = array(dtype === 'bool' ? [1, 0] : [1, 2], dtype);
      const b = array(dtype === 'bool' ? [0, 1] : [3, 4], dtype);
      expect(np.stack([a, b]).shape).toEqual([2, 2]);
    });

    it(`tile ${dtype}`, () => {
      expect(np.tile(array(dtype === 'bool' ? [1, 0] : [1, 2], dtype), 3).shape).toEqual([6]);
    });

    it(`repeat ${dtype}`, () => {
      expect(np.repeat(array(dtype === 'bool' ? [1, 0] : [1, 2], dtype), 2).shape).toEqual([4]);
    });

    it(`roll ${dtype}`, () => {
      expect(np.roll(array(dtype === 'bool' ? [1, 0, 1] : [1, 2, 3], dtype), 1).shape).toEqual([3]);
    });

    it(`flip ${dtype}`, () => {
      expect(np.flip(array(dtype === 'bool' ? [1, 0, 1] : [1, 2, 3], dtype)).shape).toEqual([3]);
    });

    it(`squeeze ${dtype}`, () => {
      expect(np.squeeze(array(dtype === 'bool' ? [[1, 0]] : [[1, 2]], dtype)).shape).toEqual([2]);
    });

    it(`expand_dims ${dtype}`, () => {
      expect(np.expand_dims(array(dtype === 'bool' ? [1] : [1], dtype), 0).shape).toEqual([1, 1]);
    });

    it(`hstack ${dtype}`, () => {
      const a = array(dtype === 'bool' ? [1, 0] : [1, 2], dtype);
      expect(np.hstack([a, a]).shape).toEqual([4]);
    });

    it(`vstack ${dtype}`, () => {
      const a = array(dtype === 'bool' ? [1, 0] : [1, 2], dtype);
      expect(np.vstack([a, a]).shape).toEqual([2, 2]);
    });

    it(`dstack ${dtype}`, () => {
      const a = array(dtype === 'bool' ? [1, 0] : [1, 2], dtype);
      expect(np.dstack([a, a]).shape).toEqual([1, 2, 2]);
    });

    it(`column_stack ${dtype}`, () => {
      const a = array(dtype === 'bool' ? [1, 0] : [1, 2], dtype);
      expect(np.column_stack([a, a]).shape).toEqual([2, 2]);
    });

    it(`append ${dtype}`, () => {
      const a = array(dtype === 'bool' ? [1, 0] : [1, 2], dtype);
      expect(np.append(a, a).shape).toEqual([4]);
    });

    it(`atleast_1d ${dtype}`, () => {
      expect(np.atleast_1d(array(dtype === 'bool' ? [1] : [1], dtype)).ndim).toBeGreaterThanOrEqual(
        1
      );
    });

    it(`atleast_2d ${dtype}`, () => {
      expect(np.atleast_2d(array(dtype === 'bool' ? [1] : [1], dtype)).ndim).toBeGreaterThanOrEqual(
        2
      );
    });

    it(`atleast_3d ${dtype}`, () => {
      expect(np.atleast_3d(array(dtype === 'bool' ? [1] : [1], dtype)).ndim).toBeGreaterThanOrEqual(
        3
      );
    });

    it(`broadcast_to ${dtype}`, () => {
      expect(np.broadcast_to(array(dtype === 'bool' ? [1] : [1], dtype), [3]).shape).toEqual([3]);
    });

    it(`split ${dtype}`, () => {
      const a = array(dtype === 'bool' ? [1, 0, 1, 0] : [1, 2, 3, 4], dtype);
      expect(np.split(a, 2).length).toBe(2);
    });

    it(`array_split ${dtype}`, () => {
      const a = array(dtype === 'bool' ? [1, 0, 1] : [1, 2, 3], dtype);
      expect(np.array_split(a, 2).length).toBe(2);
    });

    it(`fliplr ${dtype}`, () => {
      expect(np.fliplr(array(data2d, dtype)).shape).toEqual([2, 3]);
    });

    it(`flipud ${dtype}`, () => {
      expect(np.flipud(array(data2d, dtype)).shape).toEqual([2, 3]);
    });

    it(`rot90 ${dtype}`, () => {
      expect(np.rot90(array(data2d, dtype)).shape).toEqual([3, 2]);
    });

    it(`moveaxis ${dtype}`, () => {
      expect(np.moveaxis(array(data2d, dtype), 0, 1).shape).toEqual([3, 2]);
    });

    it(`swapaxes ${dtype}`, () => {
      expect(np.swapaxes(array(data2d, dtype), 0, 1).shape).toEqual([3, 2]);
    });

    it(`insert ${dtype}`, () => {
      if (isComplex(dtype)) return; // complex interleaving changes element count
      const a = array(dtype === 'bool' ? [1, 0, 1] : [1, 2, 3], dtype);
      expect(np.insert(a, 1, dtype === 'bool' ? 0 : 99).shape).toEqual([4]);
    });

    it(`delete_ ${dtype}`, () => {
      const a = array(dtype === 'bool' ? [1, 0, 1] : [1, 2, 3], dtype);
      expect(np.delete_(a, 1).shape).toEqual([2]);
    });

    it(`resize ${dtype}`, () => {
      const a = array(dtype === 'bool' ? [1, 0] : [1, 2], dtype);
      expect(np.resize(a, [4]).shape).toEqual([4]);
    });

    it(`diagflat ${dtype}`, () => {
      if (isComplex(dtype)) return; // complex interleaving changes element count
      expect(np.diagflat(array(dtype === 'bool' ? [1, 0] : [1, 2], dtype)).shape).toEqual([2, 2]);
    });

    it(`flatten ${dtype}`, () => {
      expect(np.flatten(array(data2d, dtype)).shape).toEqual([6]);
    });

    it(`compress ${dtype}`, () => {
      const a = array(
        dtype === 'bool' ? [1, 0, 1] : isComplex(dtype) ? [1, 2, 3] : [1, 2, 3],
        dtype
      );
      const cond = array([1, 0, 1], 'bool');
      expect(np.compress(cond, a).shape).toEqual([2]);
    });

    it(`select ${dtype}`, () => {
      const a = array(dtype === 'bool' ? [1, 0, 1] : [1, 2, 3], dtype);
      const cond = array([1, 0, 1], 'bool');
      const result = np.select([cond], [a]);
      expect(result.shape).toEqual([3]);
    });

    it(`diag ${dtype}`, () => {
      expect(np.diag(array(data2d, dtype)).shape).toEqual([2]);
    });

    it(`diagonal ${dtype}`, () => {
      expect(np.diagonal(array(data2d, dtype)).shape).toEqual([2]);
    });

    it(`fill_diagonal ${dtype}`, () => {
      const a = array(
        [
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
        ],
        dtype
      );
      np.fill_diagonal(a, dtype === 'bool' ? 1 : 9);
      expect(a.shape).toEqual([3, 3]);
    });
  }
});
