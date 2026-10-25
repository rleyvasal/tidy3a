#!/usr/bin/env bash
# Time tidy3a's backends on the reference workloads, in a pinned environment.
#
# The environment is built fresh from benchmarks/requirements.txt (exact
# versions) plus tidy3a itself, so packages installed elsewhere on this
# machine cannot change the results: xarray switches to opt_einsum when it is
# installed, for example. The environment is recorded with the results.
#
#   scripts/bench.sh            all workloads
#   scripts/bench.sh --quick    without the 10M-point workloads
#   scripts/bench.sh --fast     with the compiled kernels (tidy3a[fast]: Numba, pinned below)
#
# Real data, when present: NUSCENES_DATAROOT (nuScenes Mini) and
# TIDY3A_SAMPLES (air_temperature.nc, eraint_uvz.nc, L7_ETMs.tif).
# Needs uv (https://docs.astral.sh/uv/).
set -euo pipefail

cd "$(dirname "$0")/.."
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

args=()
fast=()
for arg in "$@"; do
  if [[ "$arg" == "--fast" ]]; then fast=("numba==0.68.0"); else args+=("$arg"); fi
done

uv venv -q --python 3.12 "$work/env"
uv pip install -q --python "$work/env/bin/python" -r benchmarks/requirements.txt
if [[ ${#fast[@]} -gt 0 ]]; then
  uv pip install -q --python "$work/env/bin/python" "${fast[@]}"
fi
uv pip install -q --python "$work/env/bin/python" --no-deps -e .
"$work/env/bin/python" benchmarks/run.py ${args[@]+"${args[@]}"}
