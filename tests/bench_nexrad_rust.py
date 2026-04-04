#!/usr/bin/env python
"""Benchmark comparing Rust vs Python NEXRAD Level2 parsing backends."""

import gzip
import os
import shutil
import tarfile
import tempfile
import time

import numpy as np
from open_radar_data import DATASETS


def time_fn(fn, n=5):
    """Run fn n times, return (mean_seconds, std_seconds)."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return np.mean(times), np.std(times)


def bench_datatree(label, filename_or_obj, n=5):
    """Benchmark open_nexradlevel2_datatree with Rust and Python backends."""
    from xradar.io.backends import nexrad_level2 as mod

    results = {}

    # --- Rust backend ---
    if mod._HAS_RUST:
        orig = mod._HAS_RUST
        mod._HAS_RUST = True

        def run_rust():
            dt = mod.open_nexradlevel2_datatree(
                filename_or_obj, reindex_angle=False
            )
            # Force load one variable to ensure data is actually read
            for key in dt.match("sweep_*"):
                ds = dt[key].ds
                for v in ds.data_vars:
                    if v not in (
                        "sweep_mode",
                        "sweep_number",
                        "prt_mode",
                        "follow_mode",
                        "sweep_fixed_angle",
                    ):
                        _ = ds[v].values
                        break
                break

        mean_r, std_r = time_fn(run_rust, n)
        results["rust"] = (mean_r, std_r)
        mod._HAS_RUST = orig
    else:
        print("  [Rust backend not available]")

    # --- Python backend ---
    saved = mod._HAS_RUST
    mod._HAS_RUST = False

    def run_python():
        dt = mod.open_nexradlevel2_datatree(
            filename_or_obj, reindex_angle=False
        )
        for key in dt.match("sweep_*"):
            ds = dt[key].ds
            for v in ds.data_vars:
                if v not in (
                    "sweep_mode",
                    "sweep_number",
                    "prt_mode",
                    "follow_mode",
                    "sweep_fixed_angle",
                ):
                    _ = ds[v].values
                    break
            break

    mean_p, std_p = time_fn(run_python, n)
    results["python"] = (mean_p, std_p)
    mod._HAS_RUST = saved

    # --- Report ---
    print(f"  {'Backend':<10} {'Mean (s)':>10} {'Std (s)':>10}")
    print(f"  {'-'*10} {'-'*10} {'-'*10}")
    if "rust" in results:
        print(f"  {'Rust':<10} {results['rust'][0]:>10.4f} {results['rust'][1]:>10.4f}")
    print(f"  {'Python':<10} {results['python'][0]:>10.4f} {results['python'][1]:>10.4f}")
    if "rust" in results:
        speedup = results["python"][0] / results["rust"][0]
        print(f"  Speedup: {speedup:.1f}x")
    print()
    return results


def main():
    print("=" * 60)
    print("NEXRAD Level2 Parser Benchmark: Rust vs Python")
    print("=" * 60)
    print()

    N = 5  # repetitions per benchmark

    # 1. Uncompressed file (KATX)
    print(f"[1/4] Uncompressed file (KATX20130717, VCP-11) — {N} runs")
    f_katx = DATASETS.fetch("KATX20130717_195021_V06")
    bench_datatree("KATX uncompressed", f_katx, n=N)

    # 2. BZ2 compressed file (KLBB)
    print(f"[2/4] BZ2 compressed file (KLBB20160601, VCP-21) — {N} runs")
    f_klbb = DATASETS.fetch("KLBB20160601_150025_V06")
    bench_datatree("KLBB BZ2", f_klbb, n=N)

    # 3. Legacy MSG1 file (KLIX)
    print(f"[3/4] Legacy MSG1 file (KLIX20050828) — {N} runs")
    fgz = DATASETS.fetch("KLIX20050828_180149.gz")
    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix="_klix")
    with gzip.open(fgz) as fin:
        shutil.copyfileobj(fin, tmpf)
    tmpf.close()
    bench_datatree("KLIX MSG1", tmpf.name, n=N)
    os.unlink(tmpf.name)

    # 4. Chunk files (KLOT)
    print(f"[4/4] Chunk files (KLOT, 55 S/I/E chunks) — {N} runs")
    archive = DATASETS.fetch("nexrad_level2_chunks_KLOT.tar.gz")
    tmpdir = tempfile.mkdtemp()
    with tarfile.open(archive) as tar:
        tar.extractall(tmpdir, filter="data")
    chunk_dir = os.path.join(tmpdir, "nexrad_chunks_KLOT")
    chunk_paths = sorted(
        [os.path.join(chunk_dir, f) for f in os.listdir(chunk_dir)]
    )
    bench_datatree("KLOT chunks", chunk_paths, n=N)
    shutil.rmtree(tmpdir)

    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
