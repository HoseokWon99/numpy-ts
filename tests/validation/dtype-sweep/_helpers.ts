/**
 * Shared helpers for dtype-sweep tests.
 */
import { expect } from 'vitest';
import { runNumPy as _runNumPy, arraysClose as _arraysClose, checkNumPyAvailable as _checkNumPyAvailable } from '../numpy-oracle';

export { ALL_DTYPES } from './_dtype-matrix';
export const runNumPy = _runNumPy;
export const arraysClose = _arraysClose;
export const checkNumPyAvailable = _checkNumPyAvailable;

export function npDtype(d: string) {
  return d === 'int64' ? 'np.int64' : d === 'uint64' ? 'np.uint64' : `np.${d}`;
}

export const isInt = (d: string) => d.startsWith('int') || d.startsWith('uint');
export const isFloat = (d: string) => d === 'float64' || d === 'float32';
export const isComplex = (d: string) => d.startsWith('complex');
export const isBool = (d: string) => d === 'bool';

/**
 * Handle operations where certain dtypes are expected to be rejected by both
 * JS and NumPy. Call this at the start of a test for known-unsupported dtype combos.
 *
 * Returns:
 * - `'both-reject'`   — both JS and NumPy error → caller should `return` (test passes)
 * - `'both-succeed'`  — both succeed → caller should continue with value comparison
 * - `'js-permissive'` — NumPy rejects but JS succeeds → caller MUST handle explicitly
 * - throws            — JS rejects but NumPy succeeds (genuine bug)
 *
 * @param reason  Human-readable explanation of WHY this dtype is rejected
 * @param jsFn    Thunk that runs the JS operation
 * @param pyCode  Python code for the NumPy oracle
 */
export function expectBothReject(
  reason: string,
  jsFn: () => any,
  pyCode: string
): 'both-reject' | 'both-succeed' | 'js-permissive' {
  let jsErr: Error | null = null;
  let pyErr: Error | null = null;

  try {
    jsFn();
  } catch (e: any) {
    jsErr = e;
  }

  try {
    _runNumPy(pyCode);
  } catch (e: any) {
    pyErr = e;
  }

  if (jsErr && pyErr) {
    return 'both-reject';
  }

  if (!jsErr && !pyErr) {
    return 'both-succeed';
  }

  if (jsErr && !pyErr) {
    expect.unreachable(
      `JS throws but NumPy succeeds. Reason expected: ${reason}\n` +
        `JS error: ${jsErr.message?.slice(0, 150)}`
    );
  }

  // pyErr && !jsErr — we accept dtypes that NumPy rejects.
  // Caller MUST handle this explicitly (e.g., validate JS output shape, or document the exception).
  return 'js-permissive';
}
