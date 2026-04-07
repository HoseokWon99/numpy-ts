/**
 * DType Coverage Setup — tracks which dtypes flow through which public API functions.
 *
 * Two interception points:
 * 1. ArrayStorage.dtype getter — captures operations that READ dtype (math, reductions, etc.)
 * 2. ArrayStorage.empty/zeros — captures CREATION functions that set dtype
 *
 * Each worker writes to its own file (keyed by process.pid) to avoid race conditions.
 *
 * Usage: npm run test:dtype-coverage
 */
import { ArrayStorage } from '../src/common/storage';
import { writeFileSync, mkdirSync, renameSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const HITS_DIR = join(__dirname, '..', 'coverage', 'dtype-hits');
const HITS_FILE = join(HITS_DIR, `worker-${process.pid}.json`);

const hits = new Map<string, Set<string>>();

/**
 * Walk the stack to find the highest-level src/ frame (public API entry point).
 */
function parseCaller(stack: string): string | null {
  const lines = stack.split('\n');
  let best: string | null = null;
  let bestPriority = -1;

  for (let i = 2; i < Math.min(lines.length, 15); i++) {
    const line = lines[i]!;

    // src/full/index.ts — the public API (np.sin, np.add, etc.)
    let match = line.match(/at (\w+) .*\/full\/index\.ts/);
    if (match && bestPriority < 4) { best = match[1]!; bestPriority = 4; continue; }

    // src/core/linalg.ts namespace
    match = line.match(/at (?:Object\.)?(\w+) .*\/core\/linalg\.ts/);
    if (match && bestPriority < 3) { best = `linalg.${match[1]}`; bestPriority = 3; continue; }

    // src/core/linalg.ts, core/fft.ts — namespaced
    match = line.match(/at (?:Object\.)?(\w+) .*\/core\/(linalg|fft)\.ts/);
    if (match && bestPriority < 3) { best = `${match[2]}.${match[1]}`; bestPriority = 3; continue; }

    // src/core/*.ts — other modules (creation, trig, etc.)
    match = line.match(/at (?:Object\.)?(\w+) .*\/core\/(\w+)\.ts/);
    if (match && match[2] !== 'linalg' && match[2] !== 'fft' && bestPriority < 3) { best = match[1]!; bestPriority = 3; continue; }

    // src/common/ops/*.ts
    match = line.match(/at (?:Object\.)?(\w+) .*\/ops\/(\w+)\.ts/);
    if (match && bestPriority < 2) { best = `ops/${match[2]}/${match[1]}`; bestPriority = 2; continue; }

    // src/common/wasm/*.ts
    match = line.match(/at (?:Object\.)?(\w+) .*\/wasm\/([\w-]+)\.ts/);
    if (match && match[2] !== 'runtime' && bestPriority < 1) { best = `wasm/${match[2]}/${match[1]}`; bestPriority = 1; continue; }
  }

  return best;
}

function record(caller: string | null, dtype: string) {
  if (!caller || !dtype) return;
  if (!hits.has(caller)) hits.set(caller, new Set());
  hits.get(caller)!.add(dtype);
  if (++callCount % 50 === 0) flush();
}

let lastFlushSize = 0;

function flush() {
  if (hits.size === 0) return;
  const currentSize = [...hits.values()].reduce((s, set) => s + set.size, 0);
  if (currentSize === lastFlushSize) return;
  lastFlushSize = currentSize;

  const data: Record<string, string[]> = {};
  for (const [fn, dtypes] of hits) {
    data[fn] = [...dtypes].sort();
  }
  try {
    mkdirSync(HITS_DIR, { recursive: true });
    const tmp = HITS_FILE + '.tmp';
    writeFileSync(tmp, JSON.stringify(data, null, 2));
    renameSync(tmp, HITS_FILE);
  } catch {}
}

let callCount = 0;

// --- Intercept 1: dtype getter (captures reads) ---
const originalDtype = Object.getOwnPropertyDescriptor(ArrayStorage.prototype, 'dtype')!;

Object.defineProperty(ArrayStorage.prototype, 'dtype', {
  get() {
    const value = originalDtype.get!.call(this);
    record(parseCaller(new Error().stack || ''), value);
    return value;
  },
  configurable: true,
});

// --- Intercept 2: static creation methods (captures writes) ---
const origEmpty = ArrayStorage.empty.bind(ArrayStorage);
(ArrayStorage as any).empty = function(shape: number[], dtype: string) {
  const result = origEmpty(shape, dtype);
  record(parseCaller(new Error().stack || ''), dtype);
  return result;
};

const origZeros = ArrayStorage.zeros.bind(ArrayStorage);
(ArrayStorage as any).zeros = function(shape: number[], dtype: string) {
  const result = origZeros(shape, dtype);
  record(parseCaller(new Error().stack || ''), dtype);
  return result;
};

// Also intercept ArrayStorage.ones and ArrayStorage.fromData
for (const method of ['ones', 'fromData'] as const) {
  const orig = (ArrayStorage as any)[method];
  if (typeof orig === 'function') {
    const bound = orig.bind(ArrayStorage);
    (ArrayStorage as any)[method] = function(...args: any[]) {
      const result = bound(...args);
      const dtype = args[1] ?? result?.dtype;
      if (dtype) record(parseCaller(new Error().stack || ''), String(dtype));
      return result;
    };
  }
}

process.on('exit', flush);
process.on('beforeExit', flush);
