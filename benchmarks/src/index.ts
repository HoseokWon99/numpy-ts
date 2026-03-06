/**
 * Main benchmark orchestrator
 * Runs benchmarks for both Python NumPy and numpy-ts across multiple JS runtimes
 */

import * as fs from 'fs';
import * as path from 'path';
import { getBenchmarkSpecs, filterByCategory } from './specs';
import { setBenchmarkConfig } from './runner';
import { runPythonBenchmarks } from './python-runner';
import { detectRuntimes, spawnRuntimeBenchmark } from './runtime-spawner';
import {
  compareResults,
  calculateSummary,
  printResults,
  compareMultiRuntime,
  calculateMultiRuntimeSummaries,
  printMultiRuntimeResults,
} from './analysis';
import { generateHTMLReport, generateMultiRuntimeHTMLReport } from './visualization';
import { generatePNGChart, generateMultiRuntimePNGChart } from './chart-generator';
import { validateBenchmarks } from './validation';
import type {
  BenchmarkOptions,
  BenchmarkReport,
  BenchmarkTiming,
  RuntimeName,
  MultiRuntimeReport,
} from './types';

// Read version from root package.json
const packageJson = JSON.parse(
  fs.readFileSync(path.resolve(__dirname, '../../package.json'), 'utf-8')
);

async function main() {
  // Parse command line arguments
  const args = process.argv.slice(2);
  const options: BenchmarkOptions = {
    mode: 'standard',
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];

    if (arg === '--quick') {
      options.mode = 'quick';
    } else if (arg === '--standard') {
      options.mode = 'standard';
    } else if (arg === '--large') {
      options.mode = 'large';
    } else if (arg === '--category' && i + 1 < args.length) {
      options.category = args[++i];
    } else if (arg === '--output' && i + 1 < args.length) {
      options.output = args[++i];
    } else if (arg === '--single-thread') {
      options.singleThread = true;
    } else if (arg === '--runtime' && i + 1 < args.length) {
      options.runtimes = args[++i]!.split(',').map((s) => s.trim()) as RuntimeName[];
    } else if (arg === '--help' || arg === '-h') {
      printHelp();
      process.exit(0);
    }
  }

  // Configure benchmark settings based on mode
  let minSampleTimeMs: number;
  let targetSamples: number;

  if (options.mode === 'quick') {
    minSampleTimeMs = 50;
    targetSamples = 1;
    setBenchmarkConfig(minSampleTimeMs, targetSamples);
  } else if (options.mode === 'large') {
    minSampleTimeMs = 200;
    targetSamples = 3;
    setBenchmarkConfig(minSampleTimeMs, targetSamples);
  } else {
    minSampleTimeMs = 100;
    targetSamples = 5;
    setBenchmarkConfig(minSampleTimeMs, targetSamples);
  }

  console.log('NumPy vs numpy-ts Benchmark Suite\n');
  console.log(`Mode: ${options.mode}`);
  if (options.singleThread) {
    console.log('NumPy threading: single-threaded (OMP/MKL/OpenBLAS threads = 1)');
  }

  // Detect available runtimes
  const allRuntimes = await detectRuntimes();
  let selectedRuntimes = allRuntimes;

  if (options.runtimes) {
    // Filter to requested runtimes
    selectedRuntimes = [];
    for (const requested of options.runtimes) {
      const found = allRuntimes.find((r) => r.name === requested);
      if (found) {
        selectedRuntimes.push(found);
      } else {
        console.log(`Warning: ${requested} not found on PATH, skipping`);
      }
    }
    if (selectedRuntimes.length === 0) {
      console.error('No requested runtimes are available');
      process.exit(1);
    }
  }

  const runtimeStr = selectedRuntimes
    .map((r) => `${r.name} v${r.version}`)
    .join(', ');
  console.log(`Runtimes: ${runtimeStr}`);

  // Get benchmark specifications
  let specs = getBenchmarkSpecs(options.mode || 'standard');

  if (options.category) {
    console.log(`Category filter: ${options.category}`);
    specs = filterByCategory(specs, options.category);

    if (specs.length === 0) {
      console.error('No benchmarks found for category: ' + options.category);
      process.exit(1);
    }
  }

  console.log(`Total benchmarks: ${specs.length}\n`);

  try {
    // Validate correctness before benchmarking
    const nonValidatableOperations = new Set([
      'linalg_det', 'linalg_qr', 'linalg_cholesky', 'linalg_svd', 'linalg_eig', 'linalg_eigh',
      'linalg_eigvals', 'linalg_eigvalsh', 'linalg_matrix_rank', 'linalg_pinv', 'linalg_cond',
      'linalg_lstsq', 'linalg_matrix_power'
    ]);
    const validatableSpecs = specs.filter(
      (spec) =>
        spec.category !== 'bigint' &&
        spec.category !== 'io' &&
        !nonValidatableOperations.has(spec.operation)
    );
    if (validatableSpecs.length > 0) {
      console.log('Validating correctness against NumPy...');
      await validateBenchmarks(validatableSpecs);
      console.log('');
    }
    const skippedCount = specs.length - validatableSpecs.length;
    if (skippedCount > 0) {
      console.log(
        `Skipping validation for ${skippedCount} benchmarks (BigInt/IO/Complex linalg)\n`
      );
    }

    // Run Python NumPy benchmarks (once, as the baseline)
    console.log('Running Python NumPy benchmarks...');
    const { results: numpyResults, pythonVersion, numpyVersion } = await runPythonBenchmarks(
      specs,
      minSampleTimeMs,
      targetSamples,
      options.singleThread ?? false
    );

    // Run each JS runtime via subprocess
    const runtimeResultsMap = new Map<string, BenchmarkTiming[]>();
    const runtimeVersions: Record<string, string> = {};

    for (const runtime of selectedRuntimes) {
      console.log(`\nRunning numpy-ts benchmarks under ${runtime.name}...`);
      try {
        const { results, version } = await spawnRuntimeBenchmark(
          runtime.name,
          specs,
          minSampleTimeMs,
          targetSamples
        );
        runtimeResultsMap.set(runtime.name, results);
        runtimeVersions[runtime.name] = version;
      } catch (err) {
        console.error(`Warning: ${runtime.name} benchmarks failed: ${err}`);
        console.error(`Skipping ${runtime.name}\n`);
      }
    }

    if (runtimeResultsMap.size === 0) {
      console.error('All runtime benchmarks failed');
      process.exit(1);
    }

    // Determine file suffix based on mode and threading
    const modeSuffix =
      (options.mode === 'large' ? '-large' : '') + (options.singleThread ? '_single' : '');

    // Save results
    const resultsDir = path.resolve(__dirname, '../results');
    const plotsDir = path.resolve(resultsDir, 'plots');
    if (!fs.existsSync(resultsDir)) fs.mkdirSync(resultsDir, { recursive: true });
    if (!fs.existsSync(plotsDir)) fs.mkdirSync(plotsDir, { recursive: true });

    if (runtimeResultsMap.size === 1) {
      // Single runtime: use legacy BenchmarkReport format for backward compatibility
      const [runtimeName, jsResults] = [...runtimeResultsMap.entries()][0]!;
      const comparisons = compareResults(specs, numpyResults, jsResults);
      const summary = calculateSummary(comparisons);

      printResults(comparisons, summary);

      const report: BenchmarkReport = {
        timestamp: new Date().toISOString(),
        environment: {
          node_version: runtimeVersions[runtimeName] || process.version,
          python_version: pythonVersion,
          numpy_version: numpyVersion,
          numpyjs_version: packageJson.version,
        },
        results: comparisons,
        summary,
      };

      const jsonPath = options.output
        ? path.resolve(options.output)
        : path.join(resultsDir, `latest${modeSuffix}.json`);
      fs.writeFileSync(jsonPath, JSON.stringify(report, null, 2));
      console.log(`Results saved to: ${jsonPath}`);

      const historyDir = path.join(resultsDir, 'history');
      if (!fs.existsSync(historyDir)) fs.mkdirSync(historyDir, { recursive: true });
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      fs.writeFileSync(
        path.join(historyDir, `benchmark${modeSuffix}-${timestamp}.json`),
        JSON.stringify(report, null, 2)
      );

      const htmlPath = path.join(plotsDir, `latest${modeSuffix}.html`);
      generateHTMLReport(report, htmlPath);
      console.log(`HTML report saved to: ${htmlPath}`);

      const pngPath = path.join(plotsDir, `latest${modeSuffix}.png`);
      await generatePNGChart(report, pngPath);
      console.log(`PNG chart saved to: ${pngPath}`);

      console.log(`\nView report: open ${htmlPath}`);
    } else {
      // Multiple runtimes: use MultiRuntimeReport format
      const comparisons = compareMultiRuntime(specs, numpyResults, runtimeResultsMap);
      const summaries = calculateMultiRuntimeSummaries(comparisons);

      printMultiRuntimeResults(comparisons, summaries);

      const report: MultiRuntimeReport = {
        timestamp: new Date().toISOString(),
        environment: {
          python_version: pythonVersion,
          numpy_version: numpyVersion,
          numpyjs_version: packageJson.version,
          runtimes: runtimeVersions,
        },
        results: comparisons,
        summaries,
      };

      const jsonPath = options.output
        ? path.resolve(options.output)
        : path.join(resultsDir, `latest${modeSuffix}.json`);
      fs.writeFileSync(jsonPath, JSON.stringify(report, null, 2));
      console.log(`Results saved to: ${jsonPath}`);

      const historyDir = path.join(resultsDir, 'history');
      if (!fs.existsSync(historyDir)) fs.mkdirSync(historyDir, { recursive: true });
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      fs.writeFileSync(
        path.join(historyDir, `benchmark${modeSuffix}-${timestamp}.json`),
        JSON.stringify(report, null, 2)
      );

      const htmlPath = path.join(plotsDir, `latest${modeSuffix}.html`);
      generateMultiRuntimeHTMLReport(report, htmlPath);
      console.log(`HTML report saved to: ${htmlPath}`);

      const pngPath = path.join(plotsDir, `latest${modeSuffix}.png`);
      await generateMultiRuntimePNGChart(report, pngPath);
      console.log(`PNG chart saved to: ${pngPath}`);

      console.log(`\nView report: open ${htmlPath}`);
    }
  } catch (error) {
    console.error('Benchmark failed:', error);
    process.exit(1);
  }
}

function printHelp() {
  console.log(`
NumPy vs numpy-ts Benchmark Suite

Usage:
  npm run bench [options]

Options:
  --quick              Quick benchmarks (1 sample, 50ms/sample, ~2-3min)
                       Arrays: 1K, 100x100, 500x500
  --standard           Standard benchmarks (5 samples, 100ms/sample, ~5-10min, default)
                       Arrays: 1K, 100x100, 500x500
  --large              Large array benchmarks (3 samples, 200ms/sample, ~10-20min)
                       Arrays: 10K, 316x316 (~100K), 1000x1000 (1M)
  --single-thread      Force NumPy to run single-threaded (OMP/MKL/OpenBLAS)
  --runtime <list>     Comma-separated runtimes to use (default: auto-detect)
                       Values: node, deno, bun  (e.g. --runtime node,bun)
  --category <name>    Run only benchmarks in specified category
  --output <path>      Save JSON results to specified path
  --help, -h           Show this help message

Categories:
  creation             Array creation (zeros, ones, arange, etc.)
  arithmetic           Arithmetic operations (add, multiply, etc.)
  linalg               Linear algebra (matmul, transpose)
  reductions           Reductions (sum, mean, max, min)
  reshape              Reshape operations (reshape, flatten, ravel)
  io                   IO operations (parseNpy, serializeNpy, etc.)
  bigint               BigInt (int64/uint64) operations

Examples:
  npm run bench                           # Run standard benchmarks (all detected runtimes)
  npm run bench:quick                     # Run quick benchmarks
  npm run bench -- --runtime node         # Node.js only (legacy behavior)
  npm run bench -- --runtime node,bun     # Node + Bun
  npm run bench:node                      # Shorthand: Node only
  npm run bench:deno                      # Shorthand: Deno only
  npm run bench:bun                       # Shorthand: Bun only
  npm run bench -- --category linalg      # Run only linalg benchmarks
`);
}

main().catch((err) => {
  console.error('Fatal error:', err);
  process.exit(1);
});
