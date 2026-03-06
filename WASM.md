# WASM Acceleration Plan

WASM backend as an optional accelerator. Zero API duplication — same code paths, faster inner loops. Non-WASM users pay nothing. WASM size never appears in the main bundle badge.

## Architecture

Four entrypoints, two with WASM:

| Entrypoint | Exports | WASM | Tree-shakeable |
|---|---|---|---|
| `numpy-ts` | NDArray (wraps `/core`) | No | No |
| `numpy-ts/core` | Standalone functions | No | Yes |
| `numpy-ts/wasm` | NDArray (wraps `/wasm/core`) | Yes | No |
| `numpy-ts/wasm/core` | Standalone WASM functions | Yes | Yes |

```typescript
// JS-only (existing)
import { array, add } from 'numpy-ts/core';       // tree-shakeable
import { array } from 'numpy-ts';                  // full NDArray

// WASM-accelerated (new)
import { array, add } from 'numpy-ts/wasm/core';   // tree-shakeable, WASM
import { array } from 'numpy-ts/wasm';              // full NDArray, WASM
```

Users opt into WASM by changing their import path. Same API, same types, faster inner loops. Falls back to JS automatically for small arrays or unsupported ops/dtypes.

### How it works

`numpy-ts/wasm/core` re-exports everything from `/core`, overriding specific functions with WASM-accelerated wrappers:

```typescript
// src/wasm/core.ts
export * from '../core';
export { add } from './kernels/add';       // overrides core's add
export { matmul } from './kernels/matmul'; // overrides core's matmul
// un-overridden functions pass through from core as-is
```

Each WASM kernel wrapper imports its own base64 WASM binary + the JS fallback:

```typescript
// src/wasm/kernels/add.ts
import { add as jsAdd } from '../../core/arithmetic';
import { add_f64, add_f32 } from '../bins/binary.wasm';

export function add(x1, x2) {
  if (isContiguous(x1) && x1.size > THRESHOLD) {
    // WASM fast path
  }
  return jsAdd(x1, x2); // JS fallback for small arrays / non-contiguous
}
```

`numpy-ts/wasm` exports a generated NDArray class that wraps `/wasm/core` instead of `/core`. The existing NDArray generator runs twice — same template, different import path:

```typescript
// Generated: src/full/ndarray.ts (existing)
import * as core from '../core';
class NDArray {
  add(other) { return up(core.add(this, other)); }
}

// Generated: src/wasm/ndarray.ts (new — same template, different import)
import * as core from './core';   // → numpy-ts/wasm/core (WASM-accelerated)
class NDArray {
  add(other) { return up(core.add(this, other)); }
}
```

One source of truth in the generator. Two outputs. No runtime indirection, no prototype patching, no backend registry.

### Tree-shaking behavior

**`numpy-ts/wasm/core`:** If a user imports `{ add, reshape }`, the bundler pulls in:
- `add` → WASM kernel (.wasm base64) + JS fallback
- `reshape` → just the JS version from core (no WASM overhead)
- All other WASM kernels → tree-shaken away

**`numpy-ts/wasm`:** All WASM kernels included (same as how `/` includes all JS ops). Users who chose the full NDArray API already accepted no tree-shaking. WASM compilation is still lazy — base64 strings are in the bundle but `.wasm` only compiles on first call to each kernel group.

## Language: Zig

Zig over Rust for this project because:
- **Native SIMD operators** (`+`, `*` on `@Vector` types) — cleaner than Rust intrinsics for a codebase that's 80% SIMD loops
- **No `unsafe`/inner function ceremony** — pointer-to-slice conversion at the FFI boundary gives bounds checking in debug builds, zero overhead in release, without the two-function dance
- **Built-in math** (`@sqrt`, `@exp`, `@log`) — no `libm` crate or `no_std` boilerplate
- **Less build friction** — no Cargo.toml, feature flags, or crate structure; single `zig build-lib` command
- **Explicit is appropriate** — for stateless numerical kernels, Zig's "C but better" philosophy fits better than Rust's ownership model (borrow checker is a no-op here anyway)

Rust is better suited for projects with complex allocations, shared state, and ownership graphs. These kernels are stateless functions that take pointers and do math.

## Kernel Granularity & Distribution

Each WASM kernel is compiled as a **separate, small .wasm binary** per function (all dtypes for that function). Kernels are distributed as **base64-encoded strings inside .js files**, not as separate .wasm assets.

### Why base64-in-JS

- **Universal compatibility** — works in Node, Deno, Bun, browsers, and every bundler without special WASM loader plugins or config
- **Native tree-shaking** — bundlers drop unused .js modules automatically; unused kernels never enter the bundle
- **Simple npm distribution** — just .js files, no special `files` config for .wasm assets
- **~33% size overhead is negligible** — individual kernels are a few KB; base64 adds ~1KB per kernel

### Kernel file structure

```
src/wasm/
  core.ts              # Re-exports /core, overrides WASM-accelerated functions
  ndarray.ts           # Generated NDArray wrapping /wasm/core
  index.ts             # Entrypoint for numpy-ts/wasm
  kernels/
    add.ts             # WASM wrapper: add (all dtypes), falls back to JS
    sub.ts             # WASM wrapper: sub (all dtypes)
    matmul.ts          # WASM wrapper: matmul (all dtypes)
    ...
  bins/                # Generated base64 WASM .js files
    add.wasm.js        # base64-encoded .wasm + lazy sync instantiation
    sub.wasm.js
    matmul.wasm.js
    ...
  zig/                 # Zig source (one file per kernel)
    add.zig
    sub.zig
    matmul.zig
    ...
```

Each `.wasm.js` file follows this pattern:

```js
// src/wasm/bins/add.wasm.js (generated by build)
const wasmBase64 = "AGFzbQEAAAA...";
let instance;

function init() {
  if (instance) return;
  const bytes = Uint8Array.from(atob(wasmBase64), c => c.charCodeAt(0));
  instance = new WebAssembly.Instance(new WebAssembly.Module(bytes));
}

export function add_f64(a, b, out, n) { init(); instance.exports.add_f64(a, b, out, n); }
export function add_f32(a, b, out, n) { init(); instance.exports.add_f32(a, b, out, n); }
// ...
```

**Fully synchronous — no top-level await.** Lazy init on first call. Sync `WebAssembly.Module()` works fine for small per-kernel binaries (a few KB each). This avoids Safari's top-level await bugs and works in all environments.

### Zig compilation: one .wasm per kernel

```bash
# Each kernel compiles independently
zig build-lib -target wasm32-freestanding -OReleaseFast src/wasm/zig/add.zig    -o dist/wasm/add.wasm
zig build-lib -target wasm32-freestanding -OReleaseFast src/wasm/zig/matmul.zig -o dist/wasm/matmul.wasm
# ... etc
```

The build step then base64-encodes each .wasm and wraps it in a `.wasm.js` module.

### Zig kernel conventions

FFI exports convert pointers to slices at the boundary for debug-time bounds checking:

```zig
export fn add_f64(a_ptr: [*]const f64, b_ptr: [*]const f64, out_ptr: [*]f64, n: u32) void {
    const len = @as(usize, n);
    const a = a_ptr[0..len];
    const b = b_ptr[0..len];
    const out = out_ptr[0..len];
    // SIMD loop with bounds-checked slices in debug, zero-cost in release
}
```

No inner functions needed — Zig has no `unsafe` distinction. Explicit per-dtype implementations (no comptime generics for kernels) — explicitness is preferred for numerical code where SIMD lane widths and unroll factors differ by type.

## Implementation Steps

### 1. WASM core entrypoint (`src/wasm/core.ts`)

Re-export everything from `/core`, override WASM-accelerated functions:

```typescript
export * from '../core';
export { add } from './kernels/add';
export { sub } from './kernels/sub';
export { matmul } from './kernels/matmul';
// ... only override functions that have WASM kernels
```

### 2. WASM kernel wrappers (`src/wasm/kernels/*.ts`)

Each wrapper imports its WASM binary + the JS fallback, dispatches based on array size and contiguity:

```typescript
// src/wasm/kernels/add.ts
import { add as jsAdd } from '../../core/arithmetic';
import { add_f64, add_f32 } from '../bins/add.wasm';

const THRESHOLD = 256; // minimum size for WASM to be worth the call overhead

export function add(x1: NDArrayCore, x2: NDArrayCore | number): NDArrayCore {
  if (typeof x2 !== 'number' && isContiguous(x1) && isContiguous(x2) && x1.size > THRESHOLD) {
    const out = new Float64Array(x1.size);
    add_f64(x1.data, x2.data, out, x1.size);
    return fromStorage(ArrayStorage.fromData(out, [...x1.shape], x1.dtype));
  }
  return jsAdd(x1, x2); // JS handles small arrays, broadcasting, non-contiguous
}
```

### 3. Zig kernels (`src/wasm/zig/*.zig`)

One Zig file per kernel. Each file exports functions for all supported dtypes. Use `wasm-bench/zig/` implementations as the starting point.

Key ops to start with (based on bench data):
- Binary element-wise: add, sub, mul, div
- Unary element-wise: sqrt, exp, log, sin, cos
- Matmul
- Reductions: sum, min, max

### 4. Generated NDArray for WASM (`src/wasm/ndarray.ts`)

Update `scripts/generate-full.ts` to produce two NDArray classes:
- `src/full/ndarray.ts` (existing) — imports from `../core`
- `src/wasm/ndarray.ts` (new) — imports from `./core` (WASM-accelerated)

Same template, same methods, different import path. One source of truth.

### 5. WASM full entrypoint (`src/wasm/index.ts`)

```typescript
// src/wasm/index.ts
export { NDArray } from './ndarray';         // Generated, wraps /wasm/core
export { NDArrayCore } from '../common/ndarray-core';
export { Complex } from '../common/complex';
// ... re-export same types as main index.ts
```

### 6. Package config

**package.json exports:**
```jsonc
{
  ".": { /* existing — no WASM */ },
  "./core": { /* existing — no WASM */ },
  "./wasm": {
    "types": "./dist/types/wasm/index.d.ts",
    "import": "./dist/esm/wasm/index.js",
    "default": "./dist/esm/wasm/index.js"
  },
  "./wasm/core": {
    "types": "./dist/types/wasm/core.d.ts",
    "import": "./dist/esm/wasm/core.js",
    "default": "./dist/esm/wasm/core.js"
  }
}
```

**sideEffects** stays `false` — no side-effect imports needed. Users explicitly import from `/wasm` or `/wasm/core`. WASM kernels are lazy-init, not side-effectful.

### 7. Build step

Add to `build.ts`:
1. Compile each Zig kernel to `.wasm` (`zig build-lib -target wasm32-freestanding -OReleaseFast`)
2. Base64-encode each `.wasm` binary
3. Generate `.wasm.js` wrapper files (template: base64 string + lazy sync init + exports)
4. Generate WASM NDArray via updated generator script
5. Bundle WASM entrypoints via esbuild (same as other entrypoints)

### 8. Tests

- Unit tests: WASM kernel wrappers (correct results, fallback behavior)
- Integration tests: same test suite runs against `/core` and `/wasm/core`, results must match
- Bundle tests: verify `numpy-ts` bundle size unchanged (no WASM leakage)
- Tree-shaking tests: verify importing single function from `/wasm/core` only includes that kernel

### 9. Benchmarks

Extend existing benchmark suite to compare JS vs WASM backends. Leverage `wasm-bench/` work for the Zig implementations themselves.

## TODO

- [ ] Step 1: WASM core entrypoint (re-export + override pattern)
- [ ] Step 2: WASM kernel wrappers (with size threshold + JS fallback)
- [ ] Step 3: Zig kernels (start with binary ops + matmul)
- [ ] Step 4: Update generator to produce WASM NDArray variant
- [ ] Step 5: WASM full entrypoint
- [ ] Step 6: Package config (exports for `/wasm` and `/wasm/core`)
- [ ] Step 7: Build pipeline (Zig compile + base64 encode + JS wrapper gen)
- [ ] Step 8: Tests (unit, integration, bundle, tree-shaking)
- [ ] Step 9: Benchmarks (JS vs WASM comparison)
- [ ] Update `docs/ARCHITECTURE.md` with WASM backend docs
- [ ] Update `docs/API-REFERENCE.md` with `/wasm` and `/wasm/core` entrypoint usage
- [ ] Update `README.md` with WASM installation/usage instructions

## Design Decisions

**Why Zig over Rust?** Stateless WASM kernels don't benefit from Rust's borrow checker. Zig's native SIMD operators, built-in math, and zero ceremony make it more ergonomic for this domain. Rust is better suited for projects with complex ownership.

**Why separate entrypoints, not a side-effect plugin?** A global backend registry (`import 'numpy-ts/wasm'`) would pull in ALL WASM kernels regardless of which ops are used. Separate entrypoints with re-export + override give bundlers the information they need to tree-shake unused WASM kernels.

**Why two generated NDArray classes?** NDArray methods are bound to core function imports at generation time. You can't tree-shake methods off a class, and you can't swap imports at runtime. Generating two classes (one wrapping `/core`, one wrapping `/wasm/core`) from the same template avoids runtime indirection, prototype patching, or backend registries. One source of truth in the generator — zero hand-maintained duplication.

**Why per-function .wasm, not per-group?** Maximum tree-shaking granularity. If a user imports `add` from `/wasm/core`, they get the add kernel but not sub, mul, or div. Each function's WASM binary is tiny (a few KB) so the overhead of separate modules is negligible.

**Why base64-in-JS, not separate .wasm files?** Universal bundler/runtime compatibility without plugins. Tree-shakes naturally. Simple npm distribution. Size overhead is negligible for small kernels.

**Why synchronous, not top-level await?** Safari has a [WebKit bug](https://github.com/mdn/browser-compat-data/issues/20426) where simultaneous imports of modules with top-level await can deadlock. Sync `WebAssembly.Module()` works fine for small per-kernel binaries. Lazy init means WASM only compiles on first use — no upfront cost at import time.

**Why explicit per-dtype Zig functions, not comptime generics?** SIMD lane widths and unroll factors differ by type. Explicit code is easier to debug and profile for numerical kernels. The duplication is stable — once an op works, it rarely changes.

**Why `sideEffects: false` still works?** No side-effect imports needed. Users explicitly import from `/wasm` or `/wasm/core`. WASM kernel instantiation is lazy (inside function calls), not at module evaluation time.

**Why fall back to JS for small arrays?** WASM function call overhead (~50-100ns) dominates for small arrays. Below a threshold (~256 elements), the JS fast path is faster. Each WASM kernel wrapper checks size and contiguity before dispatching to WASM.
