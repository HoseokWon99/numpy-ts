#!/usr/bin/env npx tsx
/**
 * DType Coverage Report — renders the dtype coverage matrix from dtype-hits.json.
 *
 * Usage:
 *   npm run test:dtype-coverage   # collect data
 *   npx tsx scripts/dtype-coverage-report.ts   # render report
 *   npx tsx scripts/dtype-coverage-report.ts --missing   # show only gaps
 */

import { readFileSync, existsSync } from 'fs';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const HITS_FILE = join(__dirname, '..', 'coverage', 'dtype-hits.json');
const showMissingOnly = process.argv.includes('--missing');

if (!existsSync(HITS_FILE)) {
  console.error('No dtype coverage data. Run: npm run test:dtype-coverage');
  process.exit(1);
}

const data: Record<string, string[]> = JSON.parse(readFileSync(HITS_FILE, 'utf-8'));
const DTYPES = [
  'float64', 'float32', 'float16',
  'complex128', 'complex64',
  'int64', 'uint64', 'int32', 'uint32', 'int16', 'uint16', 'int8', 'uint8',
  'bool',
];

const fns = Object.keys(data).sort();
const active = DTYPES.filter(d => fns.some(f => data[f]!.includes(d)));

// Abbreviations for column headers
const abbrev = (d: string) =>
  d.replace('float', 'f').replace('complex', 'c').replace('uint', 'u').replace('int', 'i');

const colW = 5;
const nameW = 38;

if (!showMissingOnly) {
  console.log(`\nDTYPE COVERAGE REPORT — ${fns.length} functions\n`);

  const hdr = 'Function'.padEnd(nameW) + ' ' + active.map(d => abbrev(d).padStart(colW)).join(' ');
  console.log(hdr);
  console.log('-'.repeat(hdr.length));

  for (const fn of fns) {
    const dtypes = new Set(data[fn]!);
    const cols = active.map(d =>
      dtypes.has(d)
        ? `\x1b[32m${' '.repeat(colW - 1)}✓\x1b[0m`
        : `\x1b[90m${' '.repeat(colW - 1)}·\x1b[0m`
    );
    console.log(`${fn.padEnd(nameW)} ${cols.join(' ')}`);
  }

  console.log('-'.repeat(hdr.length));
}

// Summary
let hit = 0;
let total = 0;
for (const fn of fns) {
  for (const d of active) {
    total++;
    if (data[fn]!.includes(d)) hit++;
  }
}

console.log(`\nDType coverage: ${hit}/${total} pairs (${((hit / total) * 100).toFixed(1)}%)`);
console.log(`Functions: ${fns.length} | Active dtypes: ${active.length}\n`);

// Missing per dtype
console.log('Missing coverage by dtype:');
for (const d of active) {
  const missing = fns.filter(f => !data[f]!.includes(d));
  if (missing.length === 0) {
    console.log(`  \x1b[32m${d}: fully covered\x1b[0m`);
  } else if (missing.length < fns.length) {
    const pct = (((fns.length - missing.length) / fns.length) * 100).toFixed(0);
    const preview = missing.slice(0, 8).join(', ');
    const more = missing.length > 8 ? ` +${missing.length - 8} more` : '';
    console.log(`  ${d} (${pct}%): ${preview}${more}`);
  }
}

// Most-tested and least-tested functions
console.log('\nLeast-tested functions (fewest dtypes):');
const byCount = fns
  .map(f => ({ fn: f, count: data[f]!.length }))
  .sort((a, b) => a.count - b.count);
for (const { fn, count } of byCount.slice(0, 10)) {
  console.log(`  ${fn}: ${count} dtype${count === 1 ? '' : 's'} (${data[fn]!.join(', ')})`);
}
