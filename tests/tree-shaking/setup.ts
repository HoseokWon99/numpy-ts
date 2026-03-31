/**
 * Setup file for tree-shaking tests.
 * Runs as part of the test project (respects groupOrder), NOT at vitest init.
 * Rebuilds with production (ReleaseFast) WASM so bundle sizes reflect real output.
 */
import { execSync } from 'child_process';
import { resolve } from 'path';
import { beforeAll } from 'vitest';

beforeAll(() => {
  const root = resolve(__dirname, '../..');
  console.log('[tree-shaking] Running production build (ReleaseFast)...');
  execSync('npm run build', { cwd: root, stdio: 'inherit' });
  console.log('[tree-shaking] Production build complete.');
}, 300000);
