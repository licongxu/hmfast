#!/bin/bash
# Run the tSZ Cl pipeline on CPU then TPU (in two separate JAX processes so
# the platform-aware dtype selection in hmfast.jax_platform works correctly),
# then print a side-by-side timing table.
#
# Designed to be invoked on the remote TPU VM. Drives ``benchmark_cpu_vs_tpu.py``
# in the same directory. Output JSONs land in ``./bench_results/``.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$HERE/bench_results"
mkdir -p "$RESULTS_DIR"

# Make sure ``hmfast`` is importable from the source checkout shipped alongside
# this script. The remote VM never `pip install`s the package (see
# ../tpu_submission/sync_and_run.sh for the rationale).
REPO_ROOT="$(cd "$HERE/.." && pwd)"
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

echo "====================================================="
echo " hmfast tSZ Cl benchmark: CPU vs TPU"
echo " repo: $REPO_ROOT"
echo "====================================================="

# CPU run -- force JAX onto the host CPU even on a TPU VM.
echo
echo "[1/2] CPU run (JAX_PLATFORMS=cpu)..."
JAX_PLATFORMS=cpu python3 "$HERE/benchmark_cpu_vs_tpu.py" --out "$RESULTS_DIR/cpu.json"

# TPU run -- explicit JAX_PLATFORMS=tpu so this also works on a CPU-only host
# (in which case it'll error out fast, instead of silently falling back).
echo
echo "[2/2] TPU run (JAX_PLATFORMS=tpu HMFAST_JAX_ENABLE_X64=0)..."
HMFAST_JAX_ENABLE_X64=0 JAX_PLATFORMS=tpu python3 "$HERE/benchmark_cpu_vs_tpu.py" --out "$RESULTS_DIR/tpu.json"

echo
echo "====================================================="
echo " Summary"
echo "====================================================="
python3 - "$RESULTS_DIR/cpu.json" "$RESULTS_DIR/tpu.json" <<'PY'
import json
import sys

paths = sys.argv[1:]
rows = []
for p in paths:
    with open(p) as f:
        rows.append(json.load(f))

def fmt_ms(s):
    return f"{s*1000:8.2f} ms"

header = f"{'backend':<10s} {'device':<28s} {'dtype':<10s} {'compile':>12s} {'min':>12s} {'median':>12s} {'mean':>12s} {'max':>12s}   mean_cl"
print(header)
print("-" * len(header))
for r in rows:
    print(f"{r['backend']:<10s} {r['device']:<28s} {r['dtype']:<10s} "
          f"{fmt_ms(r['compile_seconds']):>12s} "
          f"{fmt_ms(r['run_seconds_min']):>12s} "
          f"{fmt_ms(r['run_seconds_median']):>12s} "
          f"{fmt_ms(r['run_seconds_mean']):>12s} "
          f"{fmt_ms(r['run_seconds_max']):>12s}   {r['cl_mean']:.4e}")

if len(rows) == 2:
    cpu = next((r for r in rows if r['backend'] == 'cpu'), None)
    tpu = next((r for r in rows if r['backend'] == 'tpu'), None)
    if cpu and tpu:
        s_med = cpu['run_seconds_median'] / tpu['run_seconds_median']
        s_min = cpu['run_seconds_min'] / tpu['run_seconds_min']
        print()
        print(f"Speedup (median): {s_med:.2f}x   |   Speedup (best):   {s_min:.2f}x")
        # Sanity: cl_mean should agree to ~few % (float32 vs float64).
        rel = abs(cpu['cl_mean'] - tpu['cl_mean']) / abs(cpu['cl_mean'])
        print(f"cl_mean relative diff (CPU vs TPU): {rel*100:.3f}%")
PY
