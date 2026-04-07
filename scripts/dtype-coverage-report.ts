#!/usr/bin/env npx tsx
/**
 * DType Coverage Report — renders the dtype coverage matrix from dtype-hits.json,
 * measured against the expected dtype support matrix.
 *
 * Usage:
 *   npm run test:dtype-coverage          # collect data + render report
 *   npx tsx scripts/dtype-coverage-report.ts            # full matrix
 *   npx tsx scripts/dtype-coverage-report.ts --missing   # gaps only
 *   npx tsx scripts/dtype-coverage-report.ts --summary   # summary only
 */

import { readFileSync, existsSync, readdirSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { EXPECTED, SETS, totalExpectedPairs } from '../tests/validation/dtype-sweep/_dtype-matrix';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const HITS_DIR = join(__dirname, '..', 'coverage', 'dtype-hits');
const showMissing = process.argv.includes('--missing');
const showSummary = process.argv.includes('--summary');

// Merge all worker files
function loadHits(): Record<string, string[]> {
  const merged: Record<string, Set<string>> = {};

  if (!existsSync(HITS_DIR)) {
    console.error('No dtype coverage data. Run: npm run test:dtype-coverage');
    process.exit(1);
  }

  const files = readdirSync(HITS_DIR).filter((f) => f.endsWith('.json'));
  if (files.length === 0) {
    console.error('No dtype coverage data. Run: npm run test:dtype-coverage');
    process.exit(1);
  }

  for (const file of files) {
    try {
      const data: Record<string, string[]> = JSON.parse(
        readFileSync(join(HITS_DIR, file), 'utf-8')
      );
      for (const [fn, dtypes] of Object.entries(data)) {
        if (!merged[fn]) merged[fn] = new Set();
        for (const d of dtypes) merged[fn].add(d);
      }
    } catch {}
  }

  const result: Record<string, string[]> = {};
  for (const [fn, dtypes] of Object.entries(merged)) {
    result[fn] = [...dtypes].sort();
  }
  return result;
}

const hits = loadHits();

const DTYPES = SETS.ALL;
const abbrev = (d: string) =>
  d.replace('float', 'f').replace('complex', 'c').replace('uint', 'u').replace('int', 'i');

const colW = 5;
const nameW = 32;

// Compute coverage against expected matrix
let totalExpected = 0;
let totalTested = 0;
let totalMissing = 0;
const missingByDtype = new Map<string, string[]>();
const missingByFn = new Map<string, string[]>();

const fns = Object.keys(EXPECTED).sort();

for (const fn of fns) {
  const expected = new Set(EXPECTED[fn]!);
  const tested = new Set(hits[fn] || []);

  for (const d of expected) {
    totalExpected++;
    if (tested.has(d)) {
      totalTested++;
    } else {
      totalMissing++;
      if (!missingByDtype.has(d)) missingByDtype.set(d, []);
      missingByDtype.get(d)!.push(fn);
      if (!missingByFn.has(fn)) missingByFn.set(fn, []);
      missingByFn.get(fn)!.push(d);
    }
  }
}

const pct = ((totalTested / totalExpected) * 100).toFixed(1);

// Summary
console.log(`\nDTYPE COVERAGE: ${totalTested}/${totalExpected} expected pairs (${pct}%)`);
console.log(`Functions in matrix: ${fns.length} | Tracked in tests: ${Object.keys(hits).length}`);
console.log(`Missing: ${totalMissing} pairs across ${missingByFn.size} functions\n`);

if (showSummary) {
  // Per-dtype summary
  for (const d of DTYPES) {
    const missing = missingByDtype.get(d) || [];
    const total = fns.filter((f) => EXPECTED[f]!.includes(d)).length;
    const tested = total - missing.length;
    const dpct = total > 0 ? ((tested / total) * 100).toFixed(0) : '—';
    const status =
      missing.length === 0 ? '\x1b[32m✓\x1b[0m' : `\x1b[31m${missing.length} missing\x1b[0m`;
    console.log(`  ${d.padEnd(12)} ${dpct.padStart(3)}% (${tested}/${total}) ${status}`);
  }
  process.exit(0);
}

if (!showMissing) {
  // Full matrix
  const hdr =
    'Function'.padEnd(nameW) + ' ' + DTYPES.map((d) => abbrev(d).padStart(colW)).join(' ');
  console.log(hdr);
  console.log('-'.repeat(hdr.length));

  for (const fn of fns) {
    const expected = new Set(EXPECTED[fn]!);
    const tested = new Set(hits[fn] || []);

    const cols = DTYPES.map((d) => {
      if (!expected.has(d)) return `\x1b[90m${' '.repeat(colW - 1)}·\x1b[0m`; // not applicable
      if (tested.has(d)) return `\x1b[32m${' '.repeat(colW - 1)}✓\x1b[0m`; // covered
      return `\x1b[31m${' '.repeat(colW - 1)}✗\x1b[0m`; // missing
    });

    console.log(`${fn.padEnd(nameW)} ${cols.join(' ')}`);
  }

  console.log('-'.repeat(hdr.length));
  console.log(`\n✓ covered  ✗ expected but missing  · not applicable`);
}

// Missing details
if (missingByFn.size > 0) {
  console.log(`\nMissing coverage by function:`);
  const sorted = [...missingByFn.entries()].sort((a, b) => b[1].length - a[1].length);
  for (const [fn, dtypes] of sorted.slice(0, 30)) {
    console.log(`  ${fn}: ${dtypes.join(', ')}`);
  }
  if (sorted.length > 30) console.log(`  ... and ${sorted.length - 30} more functions`);
}

// Functions in expected but never seen in tests
const untested = fns.filter((f) => !hits[f]);
if (untested.length > 0) {
  console.log(`\nFunctions in matrix but NEVER tested (${untested.length}):`);
  console.log(`  ${untested.join(', ')}`);
}

// Functions seen in tests but not in expected matrix
const unmatched = Object.keys(hits)
  .filter((f) => !EXPECTED[f])
  .sort();
if (unmatched.length > 0) {
  console.log(`\nFunctions tested but NOT in matrix (${unmatched.length}):`);
  console.log(`  ${unmatched.slice(0, 20).join(', ')}${unmatched.length > 20 ? '...' : ''}`);
}
