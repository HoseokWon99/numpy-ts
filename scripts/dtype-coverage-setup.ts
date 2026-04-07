/**
 * DType Coverage Setup — tracks which dtypes flow through which public API functions
 * by intercepting ArrayStorage.dtype getter and reading the call stack.
 *
 * Writes results to coverage/dtype-hits.json periodically during tests.
 * Render the report with: npx tsx scripts/dtype-coverage-report.ts
 *
 * Usage: npm run test:dtype-coverage
 */
import { ArrayStorage } from '../src/common/storage';
import { writeFileSync, readFileSync, existsSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const HITS_FILE = join(__dirname, '..', 'coverage', 'dtype-hits.json');

const hits = new Map<string, Set<string>>();

const originalDescriptor = Object.getOwnPropertyDescriptor(
  ArrayStorage.prototype,
  'dtype'
)!;

/**
 * Walk the stack to find the highest-level src/ frame (public API entry point).
 * Priority: src/full/index.ts > src/core/linalg.ts > src/core/*.ts > ops > wasm
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

    // src/core/linalg.ts namespace functions
    match = line.match(/at (?:Object\.)?(\w+) .*\/core\/linalg\.ts/);
    if (match && bestPriority < 3) { best = `linalg.${match[1]}`; bestPriority = 3; continue; }

    // src/core/*.ts — the core module wrappers
    match = line.match(/at (?:Object\.)?(\w+) .*\/core\/(\w+)\.ts/);
    if (match && bestPriority < 3) { best = match[1]!; bestPriority = 3; continue; }

    // src/common/ops/*.ts — the ops layer
    match = line.match(/at (?:Object\.)?(\w+) .*\/ops\/(\w+)\.ts/);
    if (match && bestPriority < 2) { best = `ops/${match[2]}/${match[1]}`; bestPriority = 2; continue; }

    // src/common/wasm/*.ts — WASM wrappers
    match = line.match(/at (?:Object\.)?(\w+) .*\/wasm\/([\w-]+)\.ts/);
    if (match && match[2] !== 'runtime' && bestPriority < 1) { best = `wasm/${match[2]}/${match[1]}`; bestPriority = 1; continue; }
  }

  return best;
}

function flush() {
  let existing: Record<string, string[]> = {};
  if (existsSync(HITS_FILE)) {
    try { existing = JSON.parse(readFileSync(HITS_FILE, 'utf-8')); } catch {}
  }
  for (const [fn, dtypes] of hits) {
    const prev = new Set(existing[fn] || []);
    for (const d of dtypes) prev.add(d);
    existing[fn] = [...prev].sort();
  }
  try {
    mkdirSync(dirname(HITS_FILE), { recursive: true });
    writeFileSync(HITS_FILE, JSON.stringify(existing, null, 2));
  } catch {}
}

let callCount = 0;

Object.defineProperty(ArrayStorage.prototype, 'dtype', {
  get() {
    const value = originalDescriptor.get!.call(this);
    const caller = parseCaller(new Error().stack || '');
    if (caller && value) {
      if (!hits.has(caller)) hits.set(caller, new Set());
      hits.get(caller)!.add(value);
      if (++callCount % 500 === 0) flush();
    }
    return value;
  },
  configurable: true,
});

process.on('exit', flush);
process.on('beforeExit', flush);
