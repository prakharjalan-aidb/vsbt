"""
Benchmark Report Generator

Runs a set of benchmark configs (grouped by dataset) through pgvector_suite.py
and/or vectorchord_suite.py, then builds Pareto-style recall/QPS charts --
both per-config and combined across all configs for a dataset -- and
assembles everything into a single, leadership-ready Markdown report.

Manifest format (YAML):

    title: "Vector Index Comparison"
    datasets:
      openai-500k-cos:
        configs:
          - config/openai-500k-cos/edb_vectorplus-ivfplus-707.yaml
          - config/openai-500k-cos/pgvector-m16-128.yaml
          - config/openai-500k-cos/vectorchord-60-2k.yaml
      cohere-1m-cos:
        configs:
          - config/cohere-1m-cos/edb_vectorplus-ivfplus-1k.yaml
          - { path: config/cohere-1m-cos/pgvector-m16-128.yaml, suite: pgvector_suite.py }

Each config entry is either a plain path (suite is inferred from the
filename, same convention as utils/run_benchmarks.sh) or a dict with an
explicit `suite:` override.

Usage:
    python benchmark_report.py --manifest report_manifest.yaml
    python benchmark_report.py --manifest report_manifest.yaml --skip-run   # rebuild charts/report only
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

import chart_compare
import datasets as dataset_registry

SUITE_RUNNERS = (
    (("pgvector", "edb_vectorplus", "edb-vectorplus"), "pgvector_suite.py"),
    (("vectorchord",), "vectorchord_suite.py"),
)

METRIC_LABELS = {
    "cos": "cosine similarity", "angular": "cosine similarity",
    "l2": "Euclidean (L2) distance", "euclidean": "Euclidean (L2) distance",
    "ip": "inner product", "dot": "inner product",
}


def infer_suite(config_path: str) -> str:
    """Infer the suite runner from the config filename, matching the
    case statement in utils/run_benchmarks.sh."""
    name = Path(config_path).name.lower()
    for prefixes, runner in SUITE_RUNNERS:
        if any(name.startswith(p) for p in prefixes):
            return runner
    known = [p for prefixes, _ in SUITE_RUNNERS for p in prefixes]
    raise ValueError(
        f"Cannot infer suite runner for {config_path!r}; name it starting "
        f"with one of {known}, or set an explicit 'suite:' key in the manifest."
    )


def load_test_name_and_config(path: str) -> tuple[str, dict]:
    """Every config YAML in this repo has exactly one top-level key, the
    test name, mapping to its config dict."""
    with open(path) as f:
        doc = yaml.safe_load(f)
    if not doc or len(doc) != 1:
        raise ValueError(
            f"{path}: expected exactly one top-level test name, found {len(doc or {})}"
        )
    return next(iter(doc.items()))


def parse_manifest_entries(dataset_key: str, raw_entries: list) -> list[tuple[str, str]]:
    """Return [(config_path, runner), ...] for one dataset's manifest block."""
    entries = []
    for raw in raw_entries:
        if isinstance(raw, str):
            path, runner = raw, infer_suite(raw)
        else:
            path = raw["path"]
            runner = raw.get("suite") or infer_suite(path)
        entries.append((path, runner))
    return entries


def run_config(runner: str, config_path: str, url: str, query_clients: int,
               max_queries, warmup: str, skip_add_embeddings: bool) -> bool:
    cmd = [
        sys.executable, runner,
        "-s", config_path,
        "--url", url,
        "--query-clients", str(query_clients),
        "--warmup", warmup,
    ]
    if max_queries:
        cmd += ["--max-queries", str(max_queries)]
    if skip_add_embeddings:
        cmd += ["--skip-add-embeddings"]

    print(f"\n$ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"!! {config_path} failed (exit {result.returncode}); "
              f"continuing with remaining configs.")
        return False
    return True


def dataset_description(dataset_key: str) -> str:
    info = dataset_registry.DATASETS.get(dataset_key)
    if not info:
        return f"**{dataset_key}**"
    num = info.get("num")
    dim = info.get("dim")
    metric = METRIC_LABELS.get(info.get("metric"), info.get("metric", "N/A"))
    num_str = f"{num:,}" if isinstance(num, int) else str(num)
    return f"**{dataset_key}** — {num_str} vectors, {dim} dimensions, compared by {metric}."


def sweep_summary(points: list[dict]) -> str:
    lo, hi = points[0], points[-1]
    return (
        f"recall {lo['recall']:.3f} @ {lo['qps']:.0f} qps → "
        f"recall {hi['recall']:.3f} @ {hi['qps']:.0f} qps"
    )


def build_dataset_section(dataset_key: str, entries: list[tuple[str, str]], rows: list[dict],
                           charts_dir: Path, report_dir: Path) -> str:
    series = []
    for config_path, runner in entries:
        test_name, _ = load_test_name_and_config(config_path)
        run_id = chart_compare.find_latest_run_id(rows, test_name)
        if not run_id:
            print(f"WARNING: no results found for {test_name}; omitting from report.")
            continue
        data = chart_compare.get_series_data(rows, run_id)
        if not data["points"]:
            print(f"WARNING: {test_name} (run {run_id}) has no benchmark points; omitting.")
            continue
        series.append((test_name, config_path, data))

    anchor = dataset_key
    lines = [f"## {anchor}", "", dataset_description(dataset_key), ""]

    if not series:
        lines.append("_No benchmark results available for this dataset yet._")
        return "\n".join(lines)

    combined_qps = charts_dir / f"{dataset_key}_combined_qps.png"
    combined_p99 = charts_dir / f"{dataset_key}_combined_p99.png"
    chart_compare.plot_comparison([d for _, _, d in series], combined_qps, chart_type="qps")
    chart_compare.plot_comparison([d for _, _, d in series], combined_p99, chart_type="p99")

    lines += [
        "### Configurations compared",
        "",
        "| Config | Configuration | Recall → QPS sweep | Index build | Index size |",
        "|---|---|---|---|---|",
    ]
    for test_name, config_path, data in series:
        label = chart_compare.make_series_label(data["meta"])
        sweep = sweep_summary(data["points"])
        row = next((r for r in rows if r["test_name"] == test_name), {})
        build_t = row.get("index_build_time_s", "N/A")
        build_t_str = f"{build_t}s" if build_t not in ("N/A", "", None) else "N/A"
        idx_size = row.get("index_size", "N/A")
        lines.append(f"| `{test_name}` | {label} | {sweep} | {build_t_str} | {idx_size} |")
    lines.append("")

    lines += [
        "### Pareto comparison — all configurations",
        "",
        f"![Recall vs QPS — {dataset_key}]({combined_qps.relative_to(report_dir).as_posix()})",
        "",
        f"![Recall vs P99 latency — {dataset_key}]({combined_p99.relative_to(report_dir).as_posix()})",
        "",
        "### Individual results",
        "",
    ]

    for test_name, config_path, data in series:
        indiv_qps = charts_dir / f"{dataset_key}_{test_name}_qps.png"
        indiv_p99 = charts_dir / f"{dataset_key}_{test_name}_p99.png"
        chart_compare.plot_comparison([data], indiv_qps, chart_type="qps")
        chart_compare.plot_comparison([data], indiv_p99, chart_type="p99")
        lines += [
            f"**`{test_name}`** — `{config_path}`",
            "",
            f"![Recall vs QPS — {test_name}]({indiv_qps.relative_to(report_dir).as_posix()})",
            "",
            f"![Recall vs P99 latency — {test_name}]({indiv_p99.relative_to(report_dir).as_posix()})",
            "",
        ]

    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run benchmark configs and build a Pareto comparison report")
    parser.add_argument("--manifest", required=True, help="YAML manifest of datasets -> configs")
    parser.add_argument("--url", default=None,
                         help="Postgres URL (default: manifest's db_url, else "
                              "postgresql://postgres@localhost:5432/postgres)")
    parser.add_argument("--query-clients", type=int, default=1)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--warmup", default="auto")
    parser.add_argument("--results-dir", default="./results", help="Directory containing all_results.csv")
    parser.add_argument("--output", default="./results/report", help="Output directory for the report")
    parser.add_argument("--skip-run", action="store_true",
                         help="Skip executing the suites; only rebuild charts/report from existing results")
    parser.add_argument("--assume-loaded", action="store_true",
                         help="Pass --skip-add-embeddings for every config, not just the 2nd+ per dataset")
    parser.add_argument("--title", default=None, help="Override the manifest's report title")
    return parser


def main():
    args = build_arg_parser().parse_args()

    with open(args.manifest) as f:
        manifest = yaml.safe_load(f)

    title = args.title or manifest.get("title", "Benchmark Comparison Report")
    db_url = args.url or manifest.get("db_url", "postgresql://postgres@localhost:5432/postgres")
    datasets_block = manifest["datasets"]

    parsed = {
        dataset_key: parse_manifest_entries(dataset_key, block["configs"])
        for dataset_key, block in datasets_block.items()
    }

    if not args.skip_run:
        any_failed = False
        for dataset_key, entries in parsed.items():
            print(f"\n{'=' * 60}\n  Dataset: {dataset_key}\n{'=' * 60}")
            for i, (config_path, runner) in enumerate(entries):
                skip_add_embeddings = args.assume_loaded or i > 0
                ok = run_config(
                    runner, config_path, db_url, args.query_clients,
                    args.max_queries, args.warmup, skip_add_embeddings,
                )
                any_failed = any_failed or not ok
        if any_failed:
            print("\nSome configs failed to run; the report will omit them.")

    output_dir = Path(args.output)
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    rows = chart_compare.load_csv(args.results_dir)
    if not rows:
        print(f"No results found in {args.results_dir}/all_results.csv; nothing to report.")
        return

    sections = [
        build_dataset_section(dataset_key, entries, rows, charts_dir, output_dir)
        for dataset_key, entries in parsed.items()
    ]

    toc = "\n".join(f"- [{key}](#{key.lower()})" for key in parsed)
    report = "\n\n".join([f"# {title}", "### Contents", toc, *sections])

    report_path = output_dir / "REPORT.md"
    with open(report_path, "w") as f:
        f.write(report + "\n")

    print(f"\nReport written to {report_path}")


if __name__ == "__main__":
    main()
