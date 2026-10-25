#!/usr/bin/env bash
# Time tidy3A's backends on the reference workloads, in a pinned environment.
#
# The environment is built fresh from benchmarks/requirements.txt (exact
# versions) plus tidy3A itself, so packages installed elsewhere on this
# machine cannot change the results: xarray switches to opt_einsum when it is
# installed, for example. The environment is recorded with the results.
#
#   scripts/bench.sh            all workloads
#   scripts/bench.sh --quick    without the 10M-point workloads
#
# Real data, when present: NUSCENES_DATAROOT (nuScenes Mini) and
# TIDY3A_SAMPLES (air_temperature.nc, eraint_uvz.nc, L7_ETMs.tif).
# Needs uv (https://docs.astral.sh/uv/).
set -euo pipefail

cd "$(dirname "$0")/.."
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

uv venv -q --python 3.12 "$work/env"
uv pip install -q --python "$work/env/bin/python" -r benchmarks/requirements.txt
uv pip install -q --python "$work/env/bin/python" --no-deps -e .
"$work/env/bin/python" benchmarks/run.py "$@"
