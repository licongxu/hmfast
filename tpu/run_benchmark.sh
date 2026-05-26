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
python3 - "$RESULTS_DIR/cpu.json" "$RESULTS_DIR/tpu.json" "$HERE/tsz_cpu_vs_tpu.png" "$RESULTS_DIR/summary.json" <<'PY'
import json
import sys
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

cpu_path, tpu_path, plot_path, summary_path = sys.argv[1:]

with open(cpu_path) as f:
    cpu = json.load(f)
with open(tpu_path) as f:
    tpu = json.load(f)

# ---- timing table -------------------------------------------------------
def fmt_ms(s):
    return f"{s*1000:8.2f} ms"

header = (
    f"{'backend':<10s} {'device':<28s} {'dtype':<10s} "
    f"{'compile':>12s} {'min':>12s} {'median':>12s} {'mean':>12s} {'max':>12s}   mean_cl"
)
print(header)
print("-" * len(header))
for r in (cpu, tpu):
    print(
        f"{r['backend']:<10s} {r['device']:<28s} {r['dtype']:<10s} "
        f"{fmt_ms(r['compile_seconds']):>12s} "
        f"{fmt_ms(r['run_seconds_min']):>12s} "
        f"{fmt_ms(r['run_seconds_median']):>12s} "
        f"{fmt_ms(r['run_seconds_mean']):>12s} "
        f"{fmt_ms(r['run_seconds_max']):>12s}   {r['cl_mean']:.4e}"
    )

s_med = cpu["run_seconds_median"] / tpu["run_seconds_median"]
s_min = cpu["run_seconds_min"] / tpu["run_seconds_min"]
print()
print(f"Speedup (median): {s_med:.2f}x   |   Speedup (best):   {s_min:.2f}x")

# ---- correctness checks -------------------------------------------------
print()
print("------------------- Correctness (TPU vs CPU baseline) -------------------")

# Finite + positive on both backends.
def finite_pos(r, key):
    return r[f"{key}_all_finite"] and r[f"{key}_all_positive"]

checks = []
for tag in ("cl1h", "cl2h"):
    cpu_ok = finite_pos(cpu, tag)
    tpu_ok = finite_pos(tpu, tag)
    checks.append((f"{tag}: finite & positive on CPU", cpu_ok))
    checks.append((f"{tag}: finite & positive on TPU", tpu_ok))

ell = np.asarray(cpu["ell"])
assert ell.shape == np.asarray(tpu["ell"]).shape, "ell grid mismatch"

# Tolerances reflect float32 vs float64 on a workload spanning ~5 decades.
# Anything beyond ~5% pointwise / ~2% median would indicate a real bug.
TOL_MAX = 0.05      # 5%   max pointwise relative difference
TOL_MEDIAN = 0.02   # 2%   median pointwise relative difference
TOL_MEAN = 0.02     # 2%   mean Cl difference

def reldiff(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    denom = np.where(np.abs(a) > 0, np.abs(a), 1.0)
    return np.abs(a - b) / denom

for tag in ("cl1h", "cl2h"):
    a = np.asarray(cpu[tag])
    b = np.asarray(tpu[tag])
    rd = reldiff(a, b)
    rd_max = float(rd.max())
    rd_med = float(np.median(rd))
    print(
        f"  {tag:5s}  max-rel-diff = {rd_max*100:6.3f}%   "
        f"median-rel-diff = {rd_med*100:6.3f}%   "
        f"argmax_ell = {ell[int(rd.argmax())]:.1f}"
    )
    checks.append((f"{tag}: max relative diff < {TOL_MAX*100:g}%", rd_max < TOL_MAX))
    checks.append((f"{tag}: median relative diff < {TOL_MEDIAN*100:g}%", rd_med < TOL_MEDIAN))

cl_mean_rd = abs(cpu["cl_mean"] - tpu["cl_mean"]) / abs(cpu["cl_mean"])
print(f"  cl_mean relative diff = {cl_mean_rd*100:.3f}% (tolerance {TOL_MEAN*100:g}%)")
checks.append(("cl_mean: relative diff within tolerance", cl_mean_rd < TOL_MEAN))

# Spot pk_1h / pk_2h at the same (k, z=0.5) -- these are intermediate
# physical-units objects, so any large CPU/TPU disagreement here indicates a
# numerical-precision regression that the projected Cl might hide via
# averaging.
for key in ("pk1h_spot_k_0p01_0p1_1p0_z_0p5", "pk2h_spot_k_0p01_0p1_1p0_z_0p5"):
    a = np.asarray(cpu[key])
    b = np.asarray(tpu[key])
    rd = reldiff(a, b)
    print(f"  {key}: rel diff = {rd.tolist()}")
    checks.append((f"{key}: all entries within {TOL_MAX*100:g}%", bool(np.all(rd < TOL_MAX))))

print()
print("Pass/fail:")
all_passed = True
for name, ok in checks:
    mark = "PASS" if ok else "FAIL"
    if not ok:
        all_passed = False
    print(f"  [{mark}] {name}")

# ---- comparison plot ----------------------------------------------------
cpu_cl1h = np.asarray(cpu["cl1h"])
cpu_cl2h = np.asarray(cpu["cl2h"])
tpu_cl1h = np.asarray(tpu["cl1h"])
tpu_cl2h = np.asarray(tpu["cl2h"])
prefac = ell * (ell + 1) / (2 * np.pi)
cpu_dl_tot = (cpu_cl1h + cpu_cl2h) * prefac
tpu_dl_tot = (tpu_cl1h + tpu_cl2h) * prefac

fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True,
                          gridspec_kw={"height_ratios": [3, 1]})
ax_top, ax_bot = axes
ax_top.loglog(ell, cpu_cl1h * prefac, "--", color="tab:blue", alpha=0.8,
              label="CPU 1-halo (float64)")
ax_top.loglog(ell, cpu_cl2h * prefac, ":", color="tab:green", alpha=0.8,
              label="CPU 2-halo (float64)")
ax_top.loglog(ell, cpu_dl_tot, "-", color="black", alpha=0.8, lw=2,
              label="CPU total (float64)")
ax_top.loglog(ell, tpu_cl1h * prefac, "--", color="tab:red", alpha=0.6,
              label="TPU 1-halo (float32)")
ax_top.loglog(ell, tpu_cl2h * prefac, ":", color="tab:orange", alpha=0.6,
              label="TPU 2-halo (float32)")
ax_top.loglog(ell, tpu_dl_tot, "-", color="tab:purple", alpha=0.6, lw=2,
              label="TPU total (float32)")
ax_top.set_ylabel(r"$\ell(\ell+1)C_\ell^{yy} / 2\pi$")
ax_top.set_title("tSZ Cl: CPU vs TPU correctness")
ax_top.legend(loc="upper left", fontsize=8, ncols=2)
ax_top.grid(True, alpha=0.3)

ax_bot.semilogx(ell, (tpu_cl1h - cpu_cl1h) / cpu_cl1h * 100, "--",
                color="tab:blue", label="1-halo")
ax_bot.semilogx(ell, (tpu_cl2h - cpu_cl2h) / cpu_cl2h * 100, ":",
                color="tab:green", label="2-halo")
ax_bot.semilogx(ell, ((tpu_cl1h + tpu_cl2h) - (cpu_cl1h + cpu_cl2h)) /
                     (cpu_cl1h + cpu_cl2h) * 100, "-", color="tab:purple",
                label="total")
ax_bot.axhline(0.0, color="black", lw=0.5, alpha=0.5)
ax_bot.axhline(+TOL_MAX * 100, color="red", lw=0.5, ls="--", alpha=0.5,
               label=f"±{TOL_MAX*100:g}% tolerance")
ax_bot.axhline(-TOL_MAX * 100, color="red", lw=0.5, ls="--", alpha=0.5)
ax_bot.set_xlabel(r"Multipole $\ell$")
ax_bot.set_ylabel("(TPU - CPU) / CPU  [%]")
ax_bot.legend(loc="upper left", fontsize=8)
ax_bot.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(plot_path, dpi=200, bbox_inches="tight")
print()
print(f"Comparison plot: {plot_path}")

# ---- combined summary.json (machine-readable timing + correctness) -----
def _trim(rec):
    # Drop the big per-ell arrays from the summary so it stays small enough
    # to grep. The per-run JSONs (cpu.json / tpu.json) keep the full payload.
    drop = {"cl1h", "cl2h", "ell", "run_seconds_all"}
    return {k: v for k, v in rec.items() if k not in drop}

summary = {
    "backends": {"cpu": _trim(cpu), "tpu": _trim(tpu)},
    "timing": {
        "speedup_median": s_med,
        "speedup_best": s_min,
        "compile_seconds": {"cpu": cpu["compile_seconds"], "tpu": tpu["compile_seconds"]},
        "run_seconds_median_ms": {
            "cpu": cpu["run_seconds_median"] * 1000,
            "tpu": tpu["run_seconds_median"] * 1000,
        },
        "run_seconds_min_ms": {
            "cpu": cpu["run_seconds_min"] * 1000,
            "tpu": tpu["run_seconds_min"] * 1000,
        },
    },
    "correctness": {
        "tolerances": {"max": TOL_MAX, "median": TOL_MEDIAN, "cl_mean": TOL_MEAN},
        "cl1h": {
            "max_rel_diff": float(reldiff(cpu["cl1h"], tpu["cl1h"]).max()),
            "median_rel_diff": float(np.median(reldiff(cpu["cl1h"], tpu["cl1h"]))),
        },
        "cl2h": {
            "max_rel_diff": float(reldiff(cpu["cl2h"], tpu["cl2h"]).max()),
            "median_rel_diff": float(np.median(reldiff(cpu["cl2h"], tpu["cl2h"]))),
        },
        "cl_mean_rel_diff": cl_mean_rd,
        "pk1h_spot_rel_diff": reldiff(
            cpu["pk1h_spot_k_0p01_0p1_1p0_z_0p5"],
            tpu["pk1h_spot_k_0p01_0p1_1p0_z_0p5"],
        ).tolist(),
        "pk2h_spot_rel_diff": reldiff(
            cpu["pk2h_spot_k_0p01_0p1_1p0_z_0p5"],
            tpu["pk2h_spot_k_0p01_0p1_1p0_z_0p5"],
        ).tolist(),
        "checks": [{"name": n, "ok": ok} for n, ok in checks],
        "all_passed": all_passed,
    },
}

os.makedirs(os.path.dirname(summary_path) or ".", exist_ok=True)
with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Combined summary JSON: {summary_path}")

if not all_passed:
    print()
    print("RESULT: FAIL  (at least one correctness check did not pass)")
    sys.exit(1)
print()
print("RESULT: PASS")
PY
