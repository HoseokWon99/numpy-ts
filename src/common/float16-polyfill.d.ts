/**
 * Ambient type declaration for Float16Array (TC39 proposal).
 *
 * Available natively in Chrome 127+, Firefox 129+, Safari 18.2+, Node 23+.
 * When unavailable at runtime, numpy-ts falls back to Float32Array for float16 storage.
 *
 * This declaration ensures TypeScript compiles without errors on ES2020 targets.
 */

interface Float16Array extends ArrayLike<number> {
  readonly BYTES_PER_ELEMENT: 2;
  readonly buffer: ArrayBufferLike;
  readonly byteLength: number;
  readonly byteOffset: number;
  readonly length: number;
  set(array: ArrayLike<number>, offset?: number): void;
  slice(start?: number, end?: number): Float16Array;
  subarray(begin?: number, end?: number): Float16Array;
  fill(value: number, start?: number, end?: number): this;
  copyWithin(target: number, start: number, end?: number): this;
  [index: number]: number;
}

interface Float16ArrayConstructor {
  readonly prototype: Float16Array;
  readonly BYTES_PER_ELEMENT: 2;
  new (length: number): Float16Array;
  new (array: ArrayLike<number> | ArrayBufferLike): Float16Array;
  new (buffer: ArrayBufferLike, byteOffset?: number, length?: number): Float16Array;
  from(arrayLike: ArrayLike<number>, mapfn?: (v: number, k: number) => number): Float16Array;
  of(...items: number[]): Float16Array;
}

// eslint-disable-next-line no-redeclare
declare var Float16Array: Float16ArrayConstructor | undefined;
