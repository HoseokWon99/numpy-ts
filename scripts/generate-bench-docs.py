#!/usr/bin/env python3
"""Generate a versioned performance docs page from benchmarks/results/latest.json."""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from statistics import median as calc_median
from typing import Any

CATEGORY_ORDER = [
    "creation", "arithmetic", "math", "trig", "gradient", "linalg",
    "reductions", "manipulation", "io", "indexing", "bitwise",
    "sorting", "logic", "statistics", "sets", "random", "polynomials",
    "utilities", "fft",
]

DTYPE_ORDER = [
    "float64", "float32", "int64", "uint64", "int32", "uint32",
    "int16", "uint16", "int8", "uint8", "complex128", "complex64", "bool",
]

_DTYPE_RE = re.compile(
    r"\s+(float64|float32|complex128|complex64|int64|int32|int16|int8"
    r"|uint64|uint32|uint16|uint8|bool)$"
)


def _parse_name(name: str) -> tuple[str, str | None]:
    m = _DTYPE_RE.search(name)
    return (name[: m.start()], m.group(1)) if m else (name, None)


def _benchmark_sort_key(b: dict[str, Any]) -> tuple[str, int]:
    base, dtype = _parse_name(b["name"])
    # Treat implicit (no suffix) as float64 so it sorts first in its group
    effective = dtype if dtype is not None else "float64"
    idx = DTYPE_ORDER.index(effective) if effective in DTYPE_ORDER else len(DTYPE_ORDER)
    return (base, idx)


def _compute_dtype_stats(all_benchmarks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dtype_map: dict[str, list[float]] = {}
    for b in all_benchmarks:
        _, dtype = _parse_name(b["name"])
        dtype_map.setdefault(dtype or "float64", []).append(b["ratio"])
    result = []
    for dtype in DTYPE_ORDER:
        ratios = dtype_map.get(dtype)
        if not ratios:
            continue
        result.append({
            "dtype": dtype,
            "count": len(ratios),
            "avgSlowdown": round(sum(ratios) / len(ratios), 4),
            "medianSlowdown": round(calc_median(ratios), 4),
        })
    return result


def format_timestamp(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S %Z").strip()
    except Exception:
        return ts


def normalize_report(report: dict[str, Any], runtime: str = "node") -> dict[str, Any]:
    """Normalize multi-runtime format (summaries/runtimes) to single-runtime format (summary/ratio/numpyjs)."""
    if "summary" in report:
        return report

    normalized = {
        "timestamp": report["timestamp"],
        "environment": report["environment"],
        "summary": report["summaries"][runtime],
        "results": [],
    }
    for r in report["results"]:
        rt = r["runtimes"].get(runtime)
        if rt is None:
            continue
        normalized["results"].append({
            "name": r["name"],
            "category": r["category"],
            "numpy": r["numpy"],
            "numpyjs": rt["timing"],
            "ratio": rt["ratio"],
        })
    return normalized


def build_doc(report: dict[str, Any], source_path: str) -> str:
    report = normalize_report(report)
    sorted_results = sorted(report["results"], key=lambda r: r["ratio"], reverse=True)

    category_map: dict[str, list[dict[str, Any]]] = {}
    for r in sorted_results:
        row = {
            "name": r["name"],
            "ratio": round(r["ratio"], 4),
            "numpyOps": round(r["numpy"]["ops_per_sec"], 1),
            "numpyTsOps": round(r["numpyjs"]["ops_per_sec"], 1),
        }
        category_map.setdefault(r["category"], []).append(row)

    all_benchmarks: list[dict[str, Any]] = []
    categories: list[dict[str, Any]] = []
    for name, benchmarks in category_map.items():
        benchmarks_sorted = sorted(benchmarks, key=_benchmark_sort_key)
        avg_slowdown = round(sum(b["ratio"] for b in benchmarks_sorted) / len(benchmarks_sorted), 4)
        slower_count = sum(1 for b in benchmarks_sorted if b["ratio"] >= 1)
        categories.append(
            {
                "name": name,
                "avgSlowdown": avg_slowdown,
                "count": len(benchmarks_sorted),
                "slowerCount": slower_count,
                "fasterCount": len(benchmarks_sorted) - slower_count,
                "benchmarks": benchmarks_sorted,
            }
        )
        all_benchmarks.extend(benchmarks_sorted)

    categories.sort(key=lambda c: (
        CATEGORY_ORDER.index(c["name"]) if c["name"] in CATEGORY_ORDER else len(CATEGORY_ORDER)
    ))

    data = {
        "summary": {
            "avgSlowdown": round(report["summary"]["avg_slowdown"], 4),
            "medianSlowdown": round(report["summary"]["median_slowdown"], 4),
            "bestCase": round(report["summary"]["best_case"], 4),
            "worstCase": round(report["summary"]["worst_case"], 4),
            "totalBenchmarks": report["summary"]["total_benchmarks"],
        },
        "meta": {
            "generatedAt": format_timestamp(report["timestamp"]),
            "sourceJson": source_path,
            "runtimes": report["environment"].get("runtimes") or (
                {"node": report["environment"]["node_version"]}
                if "node_version" in report["environment"] else {}
            ),
            "pythonVersion": report["environment"].get("python_version"),
            "numpyVersion": report["environment"].get("numpy_version"),
            "numpyTsVersion": report["environment"]["numpyjs_version"],
            "machine": report["environment"].get("machine"),
        },
        "categories": categories,
        "dtypeStats": _compute_dtype_stats(all_benchmarks),
    }

    template = """---
title: Performance Benchmarks
sidebarTitle: Performance
mode: "wide"
---

import { BenchmarkReport } from '/snippets/BenchmarkReport.jsx'

export const benchmarkData = __DATA__;

Latest benchmark snapshot comparing `numpy-ts` against Python NumPy. You can run your own via `npm run bench:full`. 

<Note>
Benchmarks are generated by running a suite of ~2,000 microbenchmarks covering core NumPy APIs. Each benchmark runs the same code in Python (using NumPy) and JavaScript (using `numpy-ts`), measuring execution time from the respective language to ensure a fair comparison. 
</Note>

<BenchmarkReport data={benchmarkData} />
"""

    return template.replace("__DATA__", json.dumps(data, indent=2))


def detect_latest_docs_version(repo_root: Path) -> str:
    docs_config = repo_root / "docs" / "docs.json"
    if not docs_config.exists():
        return "v1.0.x"

    try:
        data = json.loads(docs_config.read_text(encoding="utf-8"))
        versions = data.get("navigation", {}).get("versions", [])
        if versions and isinstance(versions[0], dict) and isinstance(versions[0].get("version"), str):
            return versions[0]["version"]
    except Exception:
        pass

    return "v1.0.x"


def main() -> int:
    repo_root = Path(os.getcwd())
    input_path = (repo_root / (sys.argv[1] if len(sys.argv) > 1 else "benchmarks/results/latest.json")).resolve()
    latest_version = detect_latest_docs_version(repo_root)
    default_output = f"docs/{latest_version}/guides/performance.mdx"
    output_path = (repo_root / (sys.argv[2] if len(sys.argv) > 2 else default_output)).resolve()

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    report = json.loads(input_path.read_text(encoding="utf-8"))
    has_summary = "summary" in report or "summaries" in report
    if not isinstance(report, dict) or "results" not in report or not has_summary or "environment" not in report:
        print("Invalid benchmark report format.", file=sys.stderr)
        return 1

    source_display = str(input_path.relative_to(repo_root))
    mdx = build_doc(report, source_display)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(mdx, encoding="utf-8")
    print(f"Wrote {output_path.relative_to(repo_root)} from {source_display}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
