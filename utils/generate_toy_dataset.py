"""Generate the tiny synthetic `toy-5-cos` dataset used for fast end-to-end
pipeline smoke tests (5 base vectors, 2 queries). Writes parquet files in the
same shape `_load_parquet` in datasets.py expects.
"""
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DIM = 8
BASE_IDS = [37, 12, 88, 4, 61]
QUERY_IDS = [0, 1]
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets", "toy", "5")


def main():
    rng = np.random.default_rng(seed=42)

    base_vecs = rng.normal(size=(len(BASE_IDS), DIM)).astype(np.float64)
    query_vecs = rng.normal(size=(len(QUERY_IDS), DIM)).astype(np.float64)

    os.makedirs(OUT_DIR, exist_ok=True)

    train_table = pa.table({
        "id": pa.array(BASE_IDS, type=pa.int64()),
        "emb": pa.array(base_vecs.tolist(), type=pa.list_(pa.float64())),
    })
    pq.write_table(train_table, os.path.join(OUT_DIR, "shuffle_train.parquet"))

    test_table = pa.table({
        "id": pa.array(QUERY_IDS, type=pa.int64()),
        "emb": pa.array(query_vecs.tolist(), type=pa.list_(pa.float64())),
    })
    pq.write_table(test_table, os.path.join(OUT_DIR, "test.parquet"))

    def cosine_distance(q, v):
        return 1.0 - np.dot(q, v) / (np.linalg.norm(q) * np.linalg.norm(v))

    neighbors = []
    for q in query_vecs:
        dists = [cosine_distance(q, v) for v in base_vecs]
        ranking = [BASE_IDS[i] for i in np.argsort(dists)]
        neighbors.append(ranking)

    neighbors_table = pa.table({
        "id": pa.array(QUERY_IDS, type=pa.int64()),
        "neighbors_id": pa.array(neighbors, type=pa.list_(pa.int64())),
    })
    pq.write_table(neighbors_table, os.path.join(OUT_DIR, "neighbors.parquet"))

    print(f"Wrote toy-5-cos dataset to {OUT_DIR}")


if __name__ == "__main__":
    main()
