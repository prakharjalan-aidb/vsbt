"""Generate small CI/CD subsets of the `openai-500k-cos` dataset.

Slices the first N base vectors (1k / 5k / 10k / 20k / 50k by default) out of the
local `openai-500k` parquet train file, reuses a fixed slice of its query set, and
**recomputes** cosine ground-truth neighbours against each subset (a query's true
neighbours among 500k mostly are not in the first N rows, so GT cannot be inherited).

Writes the same three-file parquet shape `_load_parquet` in datasets.py expects:
    <out-root>/<label>/shuffle_train.parquet   (id, emb)
    <out-root>/<label>/test.parquet            (id, emb)
    <out-root>/<label>/neighbors.parquet       (id, neighbors_id)

This is a one-time, offline prep tool: it needs the 4.5 GB `openai-500k` train file
present locally. The generated files are then uploaded to S3 so CI can download them.

Usage:
    python utils/generate_openai_subsets.py
    python utils/generate_openai_subsets.py --sizes 1000,5000 --num-queries 100
"""
import argparse
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SOURCE = os.path.join(REPO_ROOT, "datasets", "openai", "500k")
DEFAULT_OUT_ROOT = os.path.join(REPO_ROOT, "datasets", "openai")
DEFAULT_SIZES = [1_000, 5_000, 10_000, 20_000, 50_000]


def _size_label(n: int) -> str:
    """1000 -> '1k', 50000 -> '50k' (matches the openai-<Nk>-cos dataset names)."""
    if n % 1_000 != 0:
        raise ValueError(f"size {n} is not a multiple of 1000")
    return f"{n // 1_000}k"


def _read_first_rows(path: str, n: int) -> pa.Table:
    """Read the first `n` rows of a parquet file, preserving its arrow schema."""
    pf = pq.ParquetFile(path)
    batches = []
    remaining = n
    for batch in pf.iter_batches(batch_size=10_000):
        if remaining <= 0:
            break
        if batch.num_rows > remaining:
            batch = batch.slice(0, remaining)
        batches.append(batch)
        remaining -= batch.num_rows
    if remaining > 0:
        raise ValueError(
            f"{path} has fewer than {n} rows (short by {remaining})"
        )
    return pa.Table.from_batches(batches)


def _embs_to_numpy(table: pa.Table) -> np.ndarray:
    """Extract the `emb` list-column into a (rows, dim) float64 array."""
    return np.asarray(table.column("emb").to_pylist(), dtype=np.float64)


def _cosine_ground_truth(base_vecs, base_ids, query_vecs, depth):
    """Return, per query, the `depth` nearest base `id`s by cosine distance.

    Cosine distance ranking == descending cosine similarity of L2-normalised
    vectors, computed as a single matmul.
    """
    def _l2_normalize(a):
        norms = np.linalg.norm(a, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return a / norms

    base_norm = _l2_normalize(base_vecs)
    query_norm = _l2_normalize(query_vecs)

    sims = query_norm @ base_norm.T  # (num_queries, N)
    depth = min(depth, base_vecs.shape[0])
    # argsort ascending on -sims -> nearest (highest similarity) first
    order = np.argsort(-sims, axis=1)[:, :depth]
    base_ids = np.asarray(base_ids)
    return [base_ids[row].tolist() for row in order]


def generate_subset(train_table_full, train_embs_full, query_table, query_embs,
                    n, out_root, gt_depth):
    label = _size_label(n)
    out_dir = os.path.join(out_root, label)
    os.makedirs(out_dir, exist_ok=True)

    train_table = train_table_full.slice(0, n)
    base_ids = train_table.column("id").to_pylist()
    base_embs = train_embs_full[:n]

    pq.write_table(train_table, os.path.join(out_dir, "shuffle_train.parquet"))
    pq.write_table(query_table, os.path.join(out_dir, "test.parquet"))

    query_ids = query_table.column("id").to_pylist()
    neighbors = _cosine_ground_truth(base_embs, base_ids, query_embs, gt_depth)

    neighbors_table = pa.table({
        "id": pa.array(query_ids, type=pa.int64()),
        "neighbors_id": pa.array(neighbors, type=pa.list_(pa.int64())),
    })
    pq.write_table(neighbors_table, os.path.join(out_dir, "neighbors.parquet"))

    print(f"  openai-{label}-cos: train={n} queries={len(query_ids)} "
          f"gt_depth={len(neighbors[0])} -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=DEFAULT_SOURCE,
                        help="dir with openai-500k parquet files (default: %(default)s)")
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT,
                        help="root for generated <label>/ subset dirs (default: %(default)s)")
    parser.add_argument("--sizes", default=",".join(str(s) for s in DEFAULT_SIZES),
                        help="comma-separated base-vector counts (default: %(default)s)")
    parser.add_argument("--num-queries", type=int, default=100,
                        help="number of test queries to slice (default: %(default)s)")
    parser.add_argument("--gt-depth", type=int, default=100,
                        help="ground-truth neighbours stored per query (default: %(default)s)")
    args = parser.parse_args()

    sizes = sorted(int(s) for s in args.sizes.split(",") if s.strip())
    max_n = max(sizes)

    train_path = os.path.join(args.source, "shuffle_train.parquet")
    test_path = os.path.join(args.source, "test.parquet")
    for p in (train_path, test_path):
        if not os.path.exists(p):
            raise FileNotFoundError(f"Source file not found: {p}")

    print(f"Reading first {max_n} base vectors from {train_path} ...")
    train_table_full = _read_first_rows(train_path, max_n)
    train_embs_full = _embs_to_numpy(train_table_full)

    print(f"Reading first {args.num_queries} queries from {test_path} ...")
    query_table = _read_first_rows(test_path, args.num_queries)
    query_embs = _embs_to_numpy(query_table)

    print("Generating subsets:")
    for n in sizes:
        generate_subset(train_table_full, train_embs_full, query_table, query_embs,
                        n, args.out_root, args.gt_depth)
    print("Done.")


if __name__ == "__main__":
    main()
