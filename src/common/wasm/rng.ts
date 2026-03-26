/**
 * WASM RNG wrapper — NumPy-compatible random number generation.
 *
 * Unlike other WASM wrappers, this is always used (no JS fallback).
 * State persists in WASM globals across calls.
 */

import {
  mt19937_init,
  mt19937_genrand,
  mt19937_random_f64,
  mt19937_get_state,
  mt19937_set_state,
  seed_sequence,
  pcg64_init_from_ss,
  pcg64_bounded_uint64,
  pcg64_random_f64,
  pcg64_get_state,
  pcg64_set_state_ptr,
  standard_normal_pcg,
  standard_exponential_pcg,
  legacy_gauss,
  legacy_standard_exponential,
  legacy_gauss_reset,
  legacy_standard_gamma,
  fill_uniform_f64_mt,
  fill_uniform_f64_pcg,
  fill_standard_normal_pcg,
  fill_standard_exponential_pcg,
  fill_legacy_gauss,
  fill_legacy_standard_exponential,
  fill_rk_interval,
  fill_legacy_standard_gamma,
  fill_legacy_chisquare,
  fill_pareto,
  fill_power,
  fill_weibull,
  fill_logistic,
  fill_gumbel,
  fill_laplace,
  fill_rayleigh,
  fill_triangular,
  fill_standard_cauchy,
  fill_lognormal,
  fill_wald,
  fill_standard_t,
  fill_beta,
  fill_f,
  fill_noncentral_chisquare,
  fill_noncentral_f,
  fill_geometric,
  fill_poisson,
  fill_binomial,
  fill_negative_binomial,
  fill_hypergeometric,
  fill_logseries,
  fill_zipf,
  fill_vonmises,
  fill_randint_i64,
  fill_randint_u8,
  fill_randint_u16,
  fill_permutation,
  fill_permutation_pcg,
  fill_bounded_uint64_pcg,
} from './bins/rng.wasm';
import { ensureMemory, resetAllocator, alloc, copyOut, copyIn } from './runtime';
import type { TypedArray } from '../dtype';

// ============================================================================
// Generic bulk fill helper — eliminates boilerplate across 30+ fill functions
// ============================================================================

type TypedArrayCtor<T extends TypedArray> = new (
  buffer: ArrayBuffer,
  byteOffset: number,
  length: number
) => T;

function bulkFill<T extends TypedArray>(
  n: number,
  Ctor: TypedArrayCtor<T>,
  fn: (outPtr: number, n: number) => void
): T {
  const bpe = (Ctor as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const bytes = n * bpe;
  ensureMemory(bytes);
  resetAllocator();
  const outPtr = alloc(bytes);
  fn(outPtr, n);
  return copyOut(outPtr, n, Ctor);
}

// ============================================================================
// MT19937
// ============================================================================

export function initMT19937(seed: number): void {
  mt19937_init(seed >>> 0);
}

export function mt19937Uint32(): number {
  return mt19937_genrand();
}

export function mt19937Float64(): number {
  return mt19937_random_f64();
}

export function getMT19937State(): { mt: Uint32Array; mti: number } {
  ensureMemory(624 * 4);
  resetAllocator();
  const outPtr = alloc(624 * 4);
  const mti = mt19937_get_state(outPtr);
  const mt = copyOut(outPtr, 624, Uint32Array);
  return { mt, mti };
}

export function setMT19937State(mt: Uint32Array, mti: number): void {
  ensureMemory(624 * 4);
  resetAllocator();
  const ptr = copyIn(mt);
  mt19937_set_state(ptr, mti);
}

// ============================================================================
// PCG64
// ============================================================================

export function initPCG64FromSeed(seed: number): void {
  ensureMemory(8 * 4);
  resetAllocator();
  const outPtr = alloc(8 * 4);
  seed_sequence(seed >>> 0, outPtr, 8);
  pcg64_init_from_ss(outPtr);
}

export function pcg64Float64(): number {
  return pcg64_random_f64();
}

export function pcg64BoundedUint64(off: number, rng: number): bigint {
  return pcg64_bounded_uint64(off, rng) as unknown as bigint;
}

export function pcg64SaveState(): BigUint64Array {
  return bulkFill(6, BigUint64Array, (p, _) => pcg64_get_state(p));
}

export function pcg64RestoreState(state: BigUint64Array): void {
  ensureMemory(6 * 8);
  resetAllocator();
  const ptr = copyIn(state);
  pcg64_set_state_ptr(ptr);
}

// ============================================================================
// Scalar distribution functions (used for size=undefined paths)
// ============================================================================

export const standardNormalPCG: () => number = standard_normal_pcg;
export const standardExponentialPCG: () => number = standard_exponential_pcg;
export const legacyGauss: () => number = legacy_gauss;
export const legacyStandardExponential: () => number = legacy_standard_exponential;
export const legacyGaussReset: () => void = legacy_gauss_reset;
export const wasmLegacyStandardGamma: (shape: number) => number = legacy_standard_gamma;

// ============================================================================
// Bulk fills — float64
// ============================================================================

export const fillUniformF64MT = (n: number) => bulkFill(n, Float64Array, fill_uniform_f64_mt);
export const fillUniformF64PCG = (n: number) => bulkFill(n, Float64Array, fill_uniform_f64_pcg);
export const fillStandardNormalPCG = (n: number) =>
  bulkFill(n, Float64Array, fill_standard_normal_pcg);
export const fillStandardExponentialPCG = (n: number) =>
  bulkFill(n, Float64Array, fill_standard_exponential_pcg);
export const fillLegacyGauss = (n: number) => bulkFill(n, Float64Array, fill_legacy_gauss);
export const fillLegacyStandardExponential = (n: number) =>
  bulkFill(n, Float64Array, fill_legacy_standard_exponential);
export const fillStandardCauchy = (n: number) => bulkFill(n, Float64Array, fill_standard_cauchy);
export const fillPermutation = (n: number) => bulkFill(n, Float64Array, fill_permutation);
export const fillPermutationPCG = (n: number) => bulkFill(n, BigInt64Array, fill_permutation_pcg);

// Float64 fills with parameters — use closures to pass extra args
export const fillLegacyStandardGamma = (n: number, shape: number) =>
  bulkFill(n, Float64Array, (p, nn) => fill_legacy_standard_gamma(p, nn, shape));
export const fillLegacyChisquare = (n: number, df: number) =>
  bulkFill(n, Float64Array, (p, nn) => fill_legacy_chisquare(p, nn, df));
export const fillPareto = (n: number, a: number) =>
  bulkFill(n, Float64Array, (p, nn) => fill_pareto(p, nn, a));
export const fillPower = (n: number, a: number) =>
  bulkFill(n, Float64Array, (p, nn) => fill_power(p, nn, a));
export const fillWeibull = (n: number, a: number) =>
  bulkFill(n, Float64Array, (p, nn) => fill_weibull(p, nn, a));
export const fillLogistic = (n: number, loc: number, scale: number) =>
  bulkFill(n, Float64Array, (p, nn) => fill_logistic(p, nn, loc, scale));
export const fillGumbel = (n: number, loc: number, scale: number) =>
  bulkFill(n, Float64Array, (p, nn) => fill_gumbel(p, nn, loc, scale));
export const fillLaplace = (n: number, loc: number, scale: number) =>
  bulkFill(n, Float64Array, (p, nn) => fill_laplace(p, nn, loc, scale));
export const fillRayleigh = (n: number, scale: number) =>
  bulkFill(n, Float64Array, (p, nn) => fill_rayleigh(p, nn, scale));
export const fillTriangular = (n: number, left: number, mode: number, right: number) =>
  bulkFill(n, Float64Array, (p, nn) => fill_triangular(p, nn, left, mode, right));
export const fillLognormal = (n: number, mean: number, sigma: number) =>
  bulkFill(n, Float64Array, (p, nn) => fill_lognormal(p, nn, mean, sigma));
export const fillWald = (n: number, mean: number, scale: number) =>
  bulkFill(n, Float64Array, (p, nn) => fill_wald(p, nn, mean, scale));
export const fillStandardT = (n: number, df: number) =>
  bulkFill(n, Float64Array, (p, nn) => fill_standard_t(p, nn, df));
export const fillBeta = (n: number, a: number, b: number) =>
  bulkFill(n, Float64Array, (p, nn) => fill_beta(p, nn, a, b));
export const fillF = (n: number, dfnum: number, dfden: number) =>
  bulkFill(n, Float64Array, (p, nn) => fill_f(p, nn, dfnum, dfden));
export const fillNoncentralChisquare = (n: number, df: number, nonc: number) =>
  bulkFill(n, Float64Array, (p, nn) => fill_noncentral_chisquare(p, nn, df, nonc));
export const fillNoncentralF = (n: number, dfnum: number, dfden: number, nonc: number) =>
  bulkFill(n, Float64Array, (p, nn) => fill_noncentral_f(p, nn, dfnum, dfden, nonc));
export const fillVonmises = (n: number, mu: number, kappa: number) =>
  bulkFill(n, Float64Array, (p, nn) => fill_vonmises(p, nn, mu, kappa));

// ============================================================================
// Bulk fills — integer (i64)
// ============================================================================

export const fillGeometric = (n: number, p: number) =>
  bulkFill(n, BigInt64Array, (ptr, nn) => fill_geometric(ptr, nn, p));
export const fillPoisson = (n: number, lam: number) =>
  bulkFill(n, BigInt64Array, (ptr, nn) => fill_poisson(ptr, nn, lam));
export const fillBinomial = (n: number, trials: number, p: number) =>
  bulkFill(n, BigInt64Array, (ptr, nn) => fill_binomial(ptr, nn, trials, p));
export const fillNegativeBinomial = (n: number, nn2: number, p: number) =>
  bulkFill(n, BigInt64Array, (ptr, nn) => fill_negative_binomial(ptr, nn, nn2, p));
export const fillHypergeometric = (n: number, ngood: number, nbad: number, nsample: number) =>
  bulkFill(n, BigInt64Array, (ptr, nn) => fill_hypergeometric(ptr, nn, ngood, nbad, nsample));
export const fillLogseries = (n: number, p: number) =>
  bulkFill(n, BigInt64Array, (ptr, nn) => fill_logseries(ptr, nn, p));
export const fillZipf = (n: number, a: number) =>
  bulkFill(n, BigInt64Array, (ptr, nn) => fill_zipf(ptr, nn, a));

// ============================================================================
// Bulk fills — special (u32, u8, u16, i64 with offset)
// ============================================================================

export const fillRkInterval = (n: number, max: number) =>
  bulkFill(n, Uint32Array, (p, nn) => fill_rk_interval(p, nn, max));
export const fillRandintI64 = (n: number, max: number, low: number) =>
  bulkFill(n, BigInt64Array, (ptr, nn) => fill_randint_i64(ptr, nn, max, low));
export const fillRandintU8 = (n: number, rng: number, off: number) =>
  bulkFill(n, Uint8Array, (ptr, nn) => fill_randint_u8(ptr, nn, rng, off));
export const fillRandintU16 = (n: number, rng: number, off: number) =>
  bulkFill(n, Uint16Array, (ptr, nn) => fill_randint_u16(ptr, nn, rng, off));
export const fillBoundedUint64PCG = (n: number, off: number, rng: number) =>
  bulkFill(n, BigInt64Array, (ptr, nn) => fill_bounded_uint64_pcg(ptr, nn, off, rng));
