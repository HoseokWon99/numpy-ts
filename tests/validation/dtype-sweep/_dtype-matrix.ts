/**
 * DType Support Matrix — defines which dtypes each public API function supports.
 *
 * Used by:
 * - dtype-sweep tests to parameterize across valid dtypes
 * - dtype-coverage-report to compute accurate coverage %
 *
 * Rules follow NumPy conventions:
 * - Math functions accept any numeric type (int→float64 promotion)
 * - Shape functions accept ALL dtypes (dtype-agnostic)
 * - Bitwise functions accept integers + bool only
 * - Linalg accepts float + complex (some float-only)
 * - FFT accepts float + complex
 * - Random always outputs float64
 */

// ============================================================
// Dtype sets
// ============================================================

export const SETS = {
  /** All 13 dtypes (excluding float16 which is platform-conditional) */
  ALL: [
    'float64', 'float32',
    'complex128', 'complex64',
    'int64', 'uint64', 'int32', 'uint32', 'int16', 'uint16', 'int8', 'uint8',
    'bool',
  ],
  /** All numeric (no bool) */
  NUMERIC: [
    'float64', 'float32',
    'complex128', 'complex64',
    'int64', 'uint64', 'int32', 'uint32', 'int16', 'uint16', 'int8', 'uint8',
  ],
  /** Real numeric (no complex, no bool) */
  REAL: [
    'float64', 'float32',
    'int64', 'uint64', 'int32', 'uint32', 'int16', 'uint16', 'int8', 'uint8',
  ],
  /** Float types only */
  FLOAT: ['float64', 'float32'],
  /** Float + complex */
  FLOAT_COMPLEX: ['float64', 'float32', 'complex128', 'complex64'],
  /** All integer types */
  INTEGER: ['int64', 'uint64', 'int32', 'uint32', 'int16', 'uint16', 'int8', 'uint8'],
  /** Integer + bool (bitwise-compatible) */
  BITWISE: ['int64', 'uint64', 'int32', 'uint32', 'int16', 'uint16', 'int8', 'uint8', 'bool'],
  /** Float64 only (random, some IO) */
  FLOAT64_ONLY: ['float64'],
  /** Representative subset for non-dtype-critical functions */
  REPRESENTATIVE: ['float64', 'int32'],
} as const;

// ============================================================
// Expected dtype support per function
// ============================================================

export const EXPECTED: Record<string, readonly string[]> = {
  // ------- Unary element-wise math -------
  absolute: SETS.NUMERIC,
  negative: SETS.NUMERIC,
  positive: SETS.NUMERIC,
  sign: SETS.REAL,
  sqrt: SETS.FLOAT_COMPLEX,
  cbrt: SETS.REAL,
  square: SETS.NUMERIC,
  reciprocal: SETS.FLOAT_COMPLEX,
  exp: SETS.NUMERIC,
  exp2: SETS.NUMERIC,
  expm1: SETS.NUMERIC,
  log: SETS.FLOAT_COMPLEX,
  log2: SETS.FLOAT_COMPLEX,
  log10: SETS.FLOAT_COMPLEX,
  log1p: SETS.FLOAT_COMPLEX,
  sin: SETS.NUMERIC,
  cos: SETS.NUMERIC,
  tan: SETS.NUMERIC,
  arcsin: SETS.NUMERIC,
  arccos: SETS.NUMERIC,
  arctan: SETS.NUMERIC,
  sinh: SETS.NUMERIC,
  cosh: SETS.NUMERIC,
  tanh: SETS.NUMERIC,
  arcsinh: SETS.NUMERIC,
  arccosh: SETS.NUMERIC,
  arctanh: SETS.NUMERIC,
  degrees: SETS.REAL,
  radians: SETS.REAL,
  deg2rad: SETS.REAL,
  rad2deg: SETS.REAL,
  ceil: SETS.REAL,
  floor: SETS.REAL,
  trunc: SETS.REAL,
  rint: SETS.REAL,
  fix: SETS.REAL,
  around: SETS.REAL,
  sinc: SETS.REAL,
  i0: SETS.FLOAT,
  spacing: SETS.FLOAT,
  nextafter: SETS.FLOAT,
  signbit: SETS.REAL,
  fabs: SETS.REAL,
  nan_to_num: SETS.FLOAT,

  // ------- Binary element-wise math -------
  add: SETS.NUMERIC,
  subtract: SETS.NUMERIC,
  multiply: SETS.NUMERIC,
  divide: SETS.NUMERIC,
  floor_divide: SETS.REAL,
  mod: SETS.REAL,
  remainder: SETS.REAL,
  fmod: SETS.REAL,
  power: SETS.NUMERIC,
  float_power: SETS.NUMERIC,
  maximum: SETS.REAL,
  minimum: SETS.REAL,
  fmax: SETS.REAL,
  fmin: SETS.REAL,
  copysign: SETS.REAL,
  hypot: SETS.REAL,
  arctan2: SETS.REAL,
  logaddexp: SETS.REAL,
  logaddexp2: SETS.REAL,
  ldexp: SETS.FLOAT,
  frexp: SETS.FLOAT,
  heaviside: SETS.FLOAT,
  gcd: SETS.INTEGER,
  lcm: SETS.INTEGER,
  divmod: SETS.REAL,

  // ------- Comparisons -------
  greater: SETS.ALL,
  greater_equal: SETS.ALL,
  less: SETS.ALL,
  less_equal: SETS.ALL,
  equal: SETS.ALL,
  not_equal: SETS.ALL,
  isclose: SETS.FLOAT,
  allclose: SETS.FLOAT,
  // array_equal: dtype-agnostic, uses element comparison not dtype dispatch
  array_equiv: SETS.ALL,

  // ------- Logical -------
  logical_and: SETS.ALL,
  logical_or: SETS.ALL,
  logical_not: SETS.ALL,
  logical_xor: SETS.ALL,
  isnan: SETS.FLOAT,
  isinf: SETS.FLOAT,
  isfinite: SETS.FLOAT,
  isneginf: SETS.FLOAT,
  isposinf: SETS.FLOAT,

  // ------- Bitwise -------
  bitwise_and: SETS.BITWISE,
  bitwise_or: SETS.BITWISE,
  bitwise_xor: SETS.BITWISE,
  bitwise_not: SETS.BITWISE,
  invert: SETS.BITWISE,
  bitwise_invert: SETS.BITWISE,
  left_shift: SETS.INTEGER,
  right_shift: SETS.INTEGER,
  bitwise_left_shift: SETS.INTEGER,
  bitwise_right_shift: SETS.INTEGER,
  bitwise_count: SETS.INTEGER,
  packbits: ['uint8'],
  unpackbits: ['uint8'],

  // ------- Reductions -------
  sum: SETS.ALL,
  prod: SETS.ALL,
  mean: SETS.NUMERIC,
  std: SETS.NUMERIC,
  variance: SETS.NUMERIC,
  max: SETS.REAL,
  min: SETS.REAL,
  // amax/amin tracked under max/min (aliases)
  argmax: SETS.REAL,
  argmin: SETS.REAL,
  any: SETS.ALL,
  all: SETS.ALL,
  count_nonzero: SETS.ALL,
  ptp: SETS.REAL,
  nansum: SETS.FLOAT,
  nanmean: SETS.FLOAT,
  nanstd: SETS.FLOAT,
  nanvar: SETS.FLOAT,
  nanmax: SETS.FLOAT,
  nanmin: SETS.FLOAT,
  nanprod: SETS.FLOAT,
  nanargmax: SETS.FLOAT,
  nanargmin: SETS.FLOAT,
  median: SETS.REAL,
  nanmedian: SETS.FLOAT,
  average: SETS.REAL,
  quantile: SETS.REAL,
  nanquantile: SETS.FLOAT,
  percentile: SETS.REAL,
  nanpercentile: SETS.FLOAT,
  cumsum: SETS.NUMERIC,
  cumprod: SETS.NUMERIC,

  // ------- Sorting & searching -------
  sort: SETS.REAL,
  argsort: SETS.REAL,
  partition: SETS.REAL,
  argpartition: SETS.REAL,
  searchsorted: SETS.REAL,
  sort_complex: SETS.FLOAT_COMPLEX,
  lexsort: SETS.REAL,
  nonzero: SETS.ALL,
  argwhere: SETS.ALL,
  flatnonzero: SETS.ALL,
  where: SETS.ALL,
  extract: SETS.ALL,

  // ------- Shape manipulation -------
  reshape: SETS.ALL,
  transpose: SETS.ALL,
  ravel: SETS.ALL,
  flatten: SETS.ALL,
  squeeze: SETS.ALL,
  expand_dims: SETS.ALL,
  concatenate: SETS.ALL,
  stack: SETS.ALL,
  hstack: SETS.ALL,
  vstack: SETS.ALL,
  dstack: SETS.ALL,
  column_stack: SETS.ALL,
  // row_stack: alias for vstack, tracked under vstack
  split: SETS.ALL,
  hsplit: SETS.ALL,
  vsplit: SETS.ALL,
  dsplit: SETS.ALL,
  array_split: SETS.ALL,
  tile: SETS.ALL,
  repeat: SETS.ALL,
  roll: SETS.ALL,
  flip: SETS.ALL,
  fliplr: SETS.ALL,
  flipud: SETS.ALL,
  rot90: SETS.ALL,
  moveaxis: SETS.ALL,
  swapaxes: SETS.ALL,
  atleast_1d: SETS.ALL,
  atleast_2d: SETS.ALL,
  atleast_3d: SETS.ALL,
  broadcast_to: SETS.ALL,
  broadcast_arrays: SETS.ALL,
  pad: SETS.REAL,
  append: SETS.ALL,
  insert: SETS.ALL,
  delete_: SETS.ALL,
  trim_zeros: SETS.REAL,
  resize: SETS.ALL,
  unstack: SETS.ALL,

  // ------- Indexing -------
  take: SETS.ALL,
  take_along_axis: SETS.ALL,
  put: SETS.REAL,
  put_along_axis: SETS.REAL,
  choose: SETS.REAL,
  compress: SETS.ALL,
  select: SETS.ALL,
  diag: SETS.ALL,
  diagonal: SETS.ALL,
  diagflat: SETS.ALL,
  // tril/triu: dtype-agnostic, don't read .dtype internally
  fill_diagonal: SETS.ALL,

  // ------- Creation -------
  array: SETS.ALL,
  zeros: SETS.ALL,
  ones: SETS.ALL,
  empty: SETS.ALL,
  full: SETS.ALL,
  arange: SETS.REAL,
  linspace: SETS.FLOAT,
  logspace: SETS.FLOAT,
  geomspace: SETS.FLOAT,
  eye: SETS.ALL,
  identity: SETS.FLOAT,
  // copy: dtype-agnostic, doesn't dispatch on dtype
  asarray: SETS.ALL,
  ascontiguousarray: SETS.ALL,
  asfortranarray: SETS.ALL,

  // ------- Set operations -------
  unique: SETS.REAL,
  union1d: SETS.REAL,
  intersect1d: SETS.REAL,
  setdiff1d: SETS.REAL,
  setxor1d: SETS.REAL,
  in1d: SETS.REAL,
  isin: SETS.REAL,

  // ------- Statistics -------
  histogram: SETS.FLOAT,
  bincount: SETS.INTEGER,
  digitize: SETS.REAL,
  corrcoef: SETS.FLOAT,
  cov: SETS.FLOAT,
  correlate: SETS.REAL,
  convolve: SETS.REAL,
  trapezoid: SETS.REAL,
  diff: SETS.REAL,
  ediff1d: SETS.REAL,
  gradient: SETS.FLOAT,
  interp: SETS.FLOAT,

  // ------- Complex -------
  real: SETS.FLOAT_COMPLEX,
  imag: SETS.FLOAT_COMPLEX,
  conj: SETS.FLOAT_COMPLEX,
  // conjugate tracked under conj (alias)
  angle: ['complex128', 'complex64'],

  // ------- Clip/misc -------
  clip: SETS.REAL,

  // ------- Linear algebra (top-level) -------
  dot: SETS.NUMERIC,
  inner: SETS.NUMERIC,
  outer: SETS.NUMERIC,
  matmul: SETS.NUMERIC,
  vecdot: SETS.NUMERIC,
  matvec: SETS.NUMERIC,
  vecmat: SETS.NUMERIC,
  cross: SETS.NUMERIC,
  kron: SETS.NUMERIC,
  tensordot: SETS.NUMERIC,
  trace: SETS.ALL,

  // ------- linalg namespace -------
  'linalg.norm': SETS.FLOAT_COMPLEX,
  'linalg.det': SETS.FLOAT,
  'linalg.inv': SETS.FLOAT,
  'linalg.solve': SETS.FLOAT,
  'linalg.eig': SETS.FLOAT,
  'linalg.eigh': SETS.FLOAT,
  'linalg.eigvals': SETS.FLOAT,
  'linalg.eigvalsh': SETS.FLOAT,
  'linalg.svd': SETS.FLOAT,
  'linalg.svdvals': SETS.FLOAT,
  'linalg.cholesky': SETS.FLOAT,
  'linalg.qr': SETS.FLOAT,
  'linalg.matrix_rank': SETS.FLOAT,
  'linalg.matrix_power': SETS.FLOAT,
  'linalg.cond': SETS.FLOAT,
  'linalg.pinv': SETS.FLOAT,
  'linalg.lstsq': SETS.FLOAT,
  'linalg.slogdet': SETS.FLOAT,
  'linalg.multi_dot': SETS.FLOAT,
  'linalg.vecdot': SETS.NUMERIC,
  'linalg.cross': SETS.NUMERIC,

  // ------- fft namespace (tracked without prefix) -------
  fft: SETS.FLOAT_COMPLEX,
  ifft: SETS.FLOAT_COMPLEX,
  rfft: SETS.FLOAT,
  irfft: SETS.FLOAT_COMPLEX,
  fftfreq: SETS.FLOAT64_ONLY,
  rfftfreq: SETS.FLOAT64_ONLY,
};

// ============================================================
// Helpers
// ============================================================

/** Get expected dtypes for a function, or null if not in the matrix */
export function getExpectedDtypes(fn: string): readonly string[] | null {
  return EXPECTED[fn] ?? null;
}

/** Count total expected pairs */
export function totalExpectedPairs(): number {
  return Object.values(EXPECTED).reduce((sum, dtypes) => sum + dtypes.length, 0);
}
