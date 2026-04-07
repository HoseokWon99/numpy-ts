/**
 * Vitest config for dtype coverage analysis.
 *
 * Runs unit + validation tests with dtype tracking enabled via
 * an ArrayStorage.dtype getter interceptor. Produces coverage/dtype-hits.json.
 *
 * Usage:
 *   npm run test:dtype-coverage          # collect data
 *   npx tsx scripts/dtype-coverage-report.ts  # render report
 */

import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    include: [
      'tests/unit/**/*.test.ts',
      'tests/validation/**/*.test.ts',
    ],
    exclude: ['**/node_modules/**'],
    setupFiles: ['./scripts/dtype-coverage-setup.ts'],
  },
});
