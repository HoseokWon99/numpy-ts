/**
 * Inject the package.json version onto globalThis as __VERSION_PLACEHOLDER__.
 *
 * The built bundle (dist/esm/) has this baked in by esbuild's `define` (see
 * build.ts). For source-import test runs (no build step), src/index.ts falls
 * back to globalThis.__VERSION_PLACEHOLDER__ — this setup file populates it.
 *
 * Wired into each source-importing project as `setupFiles` in vitest.config.ts.
 * The bundle smoke test (tests/bundles/node.test.ts) separately asserts the
 * built artifact reports the real semver, not the 'dev' fallback.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const pkg = JSON.parse(readFileSync(resolve(here, '..', 'package.json'), 'utf8'));

(globalThis as { __VERSION_PLACEHOLDER__?: string }).__VERSION_PLACEHOLDER__ = pkg.version;
