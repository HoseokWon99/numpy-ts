# WASM Acceleration Plan

WASM backend as an optional accelerator plugin. Zero API duplication — same code paths, faster inner loops. Non-WASM users pay nothing.

## Architecture

```
import np from 'numpy-ts';        // Works as-is (JS backend)
import 'numpy-ts/wasm';           // Side-effect: registers WASM fast paths
// OR
import { initWasm } from 'numpy-ts/wasm';
await initWasm();                  // Explicit async init
```

The WASM module intercepts at the fast-path level (contiguous element-wise ops, matmul, reductions). All broadcasting, dtype promotion, shape validation, and stride logic stays in TypeScript. Falls back to JS automatically for unsupported ops/dtypes.

## Implementation Steps

### 1. Backend registry (`src/common/internal/backend.ts`)

```typescript
export interface ComputeBackend {
  binaryOp?(op: string, a: Float64Array, b: Float64Array, out: Float64Array): boolean;
  unaryOp?(op: string, a: Float64Array, out: Float64Array): boolean;
  matmul?(a: Float64Array, b: Float64Array, M: number, K: number, N: number, out: Float64Array): boolean;
  reduce?(op: string, a: Float64Array, size: number): number | null;
}

let activeBackend: ComputeBackend | null = null;

export function registerBackend(backend: ComputeBackend): void {
  activeBackend = backend;
}

export function getBackend(): ComputeBackend | null {
  return activeBackend;
}
```

Methods return `boolean` / `null` to signal "handled" vs "fall back to JS". Only add methods for ops where WASM actually wins (large contiguous arrays). Start minimal, expand based on benchmarks.

### 2. Intercept existing fast paths

Add backend checks in the existing fast-path functions. Example for `addArraysFast` in `src/common/ops/arithmetic.ts`:

```typescript
function addArraysFast(a: ArrayStorage, b: ArrayStorage): ArrayStorage {
  const backend = getBackend();
  if (backend?.binaryOp) {
    const out = new Float64Array(a.size);
    if (backend.binaryOp('add', a.data as Float64Array, b.data as Float64Array, out)) {
      return ArrayStorage.fromData(out, [...a.shape], a.dtype);
    }
  }
  // existing JS fallback unchanged
}
```

Same pattern for: unary ops, matmul, reductions, sort, FFT. Only touch fast-path functions — slow paths (broadcasting, non-contiguous) stay pure JS.

### 3. WASM module (Rust or Zig)

Build a `.wasm` binary exposing the hot-loop operations. Use `wasm-bench/` results to prioritize which ops to implement first. The module operates on raw `Float64Array` buffers — no NDArray awareness needed.

Key ops to start with (based on bench data):
- Binary element-wise: add, sub, mul, div
- Unary element-wise: sqrt, exp, log, sin, cos
- Matmul
- Reductions: sum, min, max

### 4. WASM entrypoint (`src/wasm.ts`)

```typescript
import { registerBackend } from './common/internal/backend';

let initialized = false;

export async function initWasm(): Promise<void> {
  if (initialized) return;
  const wasm = await loadWasmModule();
  registerBackend({
    binaryOp(op, a, b, out) { return wasm.binary_op(op, a, b, out); },
    unaryOp(op, a, out) { return wasm.unary_op(op, a, out); },
    matmul(a, b, M, K, N, out) { return wasm.matmul(a, b, M, K, N, out); },
    reduce(op, a, size) { return wasm.reduce(op, a, size); },
  });
  initialized = true;
}

// Top-level await for side-effect import: `import 'numpy-ts/wasm'`
await initWasm();
```

### 5. Package config

**package.json exports:**
```jsonc
{
  "./wasm": {
    "types": "./dist/types/wasm.d.ts",
    "import": "./dist/esm/wasm.js",
    "default": "./dist/esm/wasm.js"
  }
}
```

**sideEffects** (granular — only WASM loses tree-shaking):
```jsonc
{
  "sideEffects": [
    "./dist/esm/wasm.js"
  ]
}
```

**files** (include .wasm binary):
```jsonc
{
  "files": [
    "dist/**/*.wasm",
    // ... existing entries
  ]
}
```

### 6. Build step

Add WASM compilation to `build.ts`:
- Compile Rust/Zig to `.wasm` (target: `wasm32-unknown-unknown` or `wasm32-wasi`)
- Generate JS glue with `wasm-pack` or manual instantiation
- Output to `dist/esm/wasm/`
- Bundle the WASM entrypoint via esbuild (same as other entrypoints)

### 7. Tests

- Unit tests: backend registry (register, dispatch, fallback)
- Integration tests: same test suite runs with and without WASM backend, results must match
- Bundle tests: verify `numpy-ts` bundle size unchanged when `/wasm` not imported
- Tree-shaking tests: verify backend registry code tree-shakes out when unused

### 8. Benchmarks

Extend existing benchmark suite to compare JS vs WASM backends. Leverage `wasm-bench/` work for the WASM implementations themselves.

## TODO

- [ ] Steps 1-2: Backend registry + fast-path intercepts
- [ ] Step 3: WASM module (start with binary ops + matmul)
- [ ] Step 4: WASM entrypoint with dual import styles
- [ ] Step 5: Package config (exports, sideEffects, files)
- [ ] Step 6: Build pipeline integration
- [ ] Step 7: Tests (unit, integration, bundle, tree-shaking)
- [ ] Step 8: Benchmarks (JS vs WASM comparison)
- [ ] Update `docs/ARCHITECTURE.md` with WASM backend docs
- [ ] Update `docs/API-REFERENCE.md` with `/wasm` entrypoint usage
- [ ] Update `README.md` with WASM installation/usage instructions

## Design Decisions

**Why not a separate API?** The `/core` entrypoint already provides a tree-shakeable alternative. Adding a third API surface would fragment the library. WASM as an accelerator means one API, optionally faster.

**Why intercept at fast-path, not ArrayStorage?** Replacing the storage layer would require async iget/iset and touch every code path. Intercepting fast paths is surgical — only hot loops change, and fallback is automatic.

**Why `sideEffects` per-file?** Keeps `numpy-ts` and `numpy-ts/core` fully tree-shakeable. Only the WASM glue file (which must run on import) is marked.

**Minimum viable scope:** Binary ops + matmul. These show the largest WASM wins in benchmarks and cover the most common workloads. Expand based on profiling.
