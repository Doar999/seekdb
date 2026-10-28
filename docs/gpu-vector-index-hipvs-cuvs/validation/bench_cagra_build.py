#!/usr/bin/env python3
"""Reproducible CAGRA build benchmark for seekdb lib=cuvs vs lib=vsag.

Measures, for an already-loaded vector table:
  build_s  = CREATE VECTOR INDEX ... + the first fixed ANN query (the lazy
             CAGRA build in the first query is therefore always timed)
  recall@10 / recall@100 of APPROXIMATE queries against numpy brute force
  single-query P50/P95 latency

Each round drops the index first, so no CAGRA graph, index file or GPU index
is reused between rounds. Run it against the frozen baseline and the candidate
(alternating, cold-started observers) and compare the medians.

Requirements: python3 + numpy, a `mysql` client on PATH, a running observer.

Example:
  python3 bench_cagra_build.py --port 2881 --n 200000 --dim 128 --rounds 3 \
      --lib cuvs --out /tmp/cand_cuvs.json
"""
import argparse
import json
import pathlib
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np

DB = "cagra_bench"
TABLE = "t_bench"
INDEX = "idx_bench"


def mysql_cmd(port, db):
    cmd = ["mysql", "--protocol=TCP", "-h127.0.0.1", f"-P{port}", "-uroot", "-A", "-N", "-B"]
    if db:
        cmd.append(db)
    return cmd


def mysql(port, sql, db=DB, timeout=7200):
    # the first query of a lib=cuvs index includes the GPU CAGRA build
    sql = "set ob_query_timeout=3600000000;" + sql
    res = subprocess.run(mysql_cmd(port, db), input=sql, capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        raise RuntimeError(f"mysql failed: {res.stderr.strip()}\nSQL: {sql[:300]}")
    return res.stdout


def vec_literal(v):
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def read_fvecs(path, limit=None):
    raw = np.fromfile(path, dtype=np.int32)
    dim = raw[0]
    data = raw.reshape(-1, dim + 1)[:, 1:].view(np.float32)
    return data[:limit] if limit else data


def make_data(args):
    if args.base_fvecs:
        base = read_fvecs(args.base_fvecs, args.n)
        if args.query_fvecs:
            query = read_fvecs(args.query_fvecs, args.nq)
        else:
            query = base[: args.nq] + 0.01
    else:
        rng = np.random.default_rng(args.seed)
        base = rng.normal(size=(args.n, args.dim)).astype(np.float32)
        idx = rng.choice(args.n, size=args.nq, replace=False)
        query = base[idx] + rng.normal(scale=0.05, size=(args.nq, args.dim)).astype(np.float32)
    return (np.ascontiguousarray(base, dtype=np.float32),
            np.ascontiguousarray(query, dtype=np.float32))


def ground_truth(base, query, k):
    gt = np.empty((len(query), k), dtype=np.int64)
    bnorm = (base * base).sum(axis=1)
    for s in range(0, len(query), 256):
        q = query[s:s + 256]
        d = bnorm[None, :] - 2.0 * q @ base.T
        part = np.argpartition(d, k, axis=1)[:, :k]
        order = np.take_along_axis(d, part, axis=1).argsort(axis=1)
        gt[s:s + 256] = np.take_along_axis(part, order, axis=1)
    return gt


def load_table(args, base):
    mysql(args.port, f"create database if not exists {DB};", db=None)
    mysql(args.port, f"drop table if exists {TABLE};"
                     f"create table {TABLE}(id bigint primary key, v vector({base.shape[1]}));")
    with tempfile.NamedTemporaryFile("w", suffix=".sql", delete=False) as f:
        for s in range(0, len(base), 1000):
            rows = ",".join(f"({i},'{vec_literal(base[i])}')" for i in range(s, min(s + 1000, len(base))))
            f.write(f"insert into {TABLE} values {rows};\n")
        path = f.name
    try:
        with open(path) as f:
            res = subprocess.run(mysql_cmd(args.port, DB), stdin=f, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(res.stderr)
    finally:
        pathlib.Path(path).unlink()


def index_exists(args):
    out = mysql(args.port, "select count(*) from information_schema.statistics "
                           f"where table_schema='{DB}' and table_name='{TABLE}' "
                           f"and index_name='{INDEX}';", db=None)
    return out.strip() not in ("", "0")


def ann_ids(args, q, k):
    out = mysql(args.port, f"select id from {TABLE} order by l2_distance(v,'{vec_literal(q)}') "
                           f"approximate limit {k};")
    return [int(x) for x in out.split()]


def one_round(args, query, gt):
    if index_exists(args):
        mysql(args.port, f"drop index {INDEX} on {TABLE};")
    ddl = (f"create vector index {INDEX} on {TABLE}(v) with (distance=l2, type=hnsw, lib={args.lib}, "
           f"m={args.m}, ef_construction={args.ef_construction}, ef_search={args.ef_search});")
    t0 = time.perf_counter()
    mysql(args.port, ddl)
    first = ann_ids(args, query[0], 10)
    build_s = time.perf_counter() - t0
    if len(first) != 10:
        raise RuntimeError(f"first query returned {len(first)} rows")

    lat, hit10, hit100 = [], 0, 0
    for i, q in enumerate(query):
        t1 = time.perf_counter()
        ids = ann_ids(args, q, 100)
        lat.append(time.perf_counter() - t1)
        hit10 += len(set(ids[:10]) & set(gt[i, :10].tolist()))
        hit100 += len(set(ids) & set(gt[i].tolist()))
    lat.sort()
    return {
        "build_s": build_s,
        "recall10": hit10 / (10.0 * len(query)),
        "recall100": hit100 / (100.0 * len(query)),
        "p50_ms": 1000 * lat[len(lat) // 2],
        "p95_ms": 1000 * lat[max(int(len(lat) * 0.95) - 1, 0)],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=2881)
    p.add_argument("--lib", choices=["cuvs", "vsag"], default="cuvs")
    p.add_argument("--n", type=int, default=100000)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--nq", type=int, default=200)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--m", type=int, default=16)
    p.add_argument("--ef-construction", dest="ef_construction", type=int, default=200)
    p.add_argument("--ef-search", dest="ef_search", type=int, default=200)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--base-fvecs")
    p.add_argument("--query-fvecs")
    p.add_argument("--skip-load", action="store_true", help="table already holds the same data")
    p.add_argument("--out", default="-")
    args = p.parse_args()

    base, query = make_data(args)
    gt = ground_truth(base, query, 100)
    if not args.skip_load:
        load_table(args, base)
    rounds = [one_round(args, query, gt) for _ in range(args.rounds)]
    summary = {
        "lib": args.lib, "n": len(base), "dim": int(base.shape[1]), "nq": len(query),
        "rounds": rounds,
        "build_s_median": statistics.median(r["build_s"] for r in rounds),
        "recall10_min": min(r["recall10"] for r in rounds),
        "recall100_min": min(r["recall100"] for r in rounds),
    }
    text = json.dumps(summary, indent=2)
    if args.out != "-":
        pathlib.Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
