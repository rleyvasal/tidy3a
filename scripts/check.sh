#!/usr/bin/env bash
# Run what CI runs (.github/workflows/tests.yml), locally, before pushing.
#
# Each job gets a fresh environment with only tidy3a's declared
# dependencies, so nothing installed on this machine (tidy3, plot3, a newer
# xarray) can hide a failure that CI would see.
#
#   scripts/check.sh              all jobs, one after another
#   scripts/check.sh 3.12         one Python version only; 3.14t for the free-threaded build, which installs the
#                                 test extra (dev without Polars, which has no free-threaded wheels) and runs the
#                                 tests with the GIL off, after checking that importing tidy3a and what it uses leaves
#                                 it off
#   scripts/check.sh --parallel   all jobs side by side, for a machine with cores to spare: the environments are
#                                 set up one after another (each install builds tidy3a in this folder), each job's
#                                 tests start once its environment is ready, the distribution job once every
#                                 install is done, and each job's output is shown, in order, when all have finished
#
# Needs uv (https://docs.astral.sh/uv/); it downloads missing Pythons.
set -euo pipefail

parallel=0
if [[ "${1:-}" == "--parallel" ]]; then
  parallel=1
  shift
fi

cd "$(dirname "$0")/.."
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

versions=("${@:-3.11 3.12 3.13 3.14 3.14t}")
versions=(${versions[@]})
failed=()
names=()  # the jobs started side by side, and their output
logs=()

run() {
  local name="$1"; shift
  if (( parallel )); then
    local log="$work/job-${#names[@]}.log"
    names+=("$name")
    logs+=("$log")
    ( set +e; "$@" > "$log" 2>&1; echo $? > "$log.status" ) &
    return 0
  fi
  printf '\n── %s ──\n' "$name"
  if "$@"; then
    printf '   ok: %s\n' "$name"
  else
    printf '   FAILED: %s\n' "$name"
    failed+=("$name")
  fi
}

setup() {
  local env="$1"; shift
  local python="$1"; shift
  [[ "$python" == *t ]] || python="$python+gil"  # uv may otherwise take an installed free-threaded build for 3.14
  uv venv -q --python "$python" "$env" &&
    uv pip install -q --python "$env/bin/python" "$@"
}

tests() {
  [[ -x "$1/bin/python" ]] || { echo "its environment could not be set up"; return 1; }
  if "$1/bin/python" -c 'import sys, sysconfig; sys.exit(not sysconfig.get_config_var("Py_GIL_DISABLED"))'; then
    "$1/bin/python" -c 'import sys, numba, tidy3a, xarray, zarr; from tidy3a.fused import driver
assert not sys._is_gil_enabled(), "a module imported turned the GIL on"' &&
      "$1/bin/python" -X gil=0 -m pytest -q -p no:cacheprovider
  else
    "$1/bin/python" -m pytest -q -p no:cacheprovider
  fi
}

pytest_job() {
  local env="$work/py$1-$RANDOM"
  setup "$env" "$@" && tests "$env"
}

job() {
  local name="$1"; shift
  if (( parallel )); then
    local env="$work/py$1-$RANDOM"
    setup "$env" "$@" || rm -rf "$env"  # now, so no two installs build tidy3a at once
    run "$name" tests "$env"
  else
    run "$name" pytest_job "$@"
  fi
}

for python in "${versions[@]}"; do
  if [[ "$python" == *t ]]; then
    job "pytest (Python ${python%t}, free-threaded)" "$python" -e ".[test]"
  else
    job "pytest (Python $python)" "$python" -e ".[dev]"
  fi
done

if [[ $# -eq 0 ]]; then
  job "pytest (minimum NumPy and xarray)" \
    3.11 "numpy==2.0.*" "pandas==2.2.2" "xarray==2024.6.0" "dask[array]" pytest -e .

  distribution_job() {
    uv build -q --out-dir "$work/dist" &&
      uvx twine check --strict "$work"/dist/* &&
      uv venv -q --python 3.12 "$work/clean" &&
      uv pip install -q --python "$work/clean/bin/python" "$work"/dist/*.whl &&
      (cd "$work" && "$work/clean/bin/python" -c \
        "import tidy3a, numpy as np; print(tidy3a.read_array(np.zeros(3), dims=['t']).to_dict())") &&
      uv venv -q --python 3.12 "$work/full" &&
      uv pip install -q --python "$work/full/bin/python" "$(ls "$work"/dist/*.whl)[xarray]" &&
      (cd "$work" && "$work/full/bin/python" -c \
        "import tidy3a, xarray as xr; print(tidy3a.tidy(xr.Dataset({'a': ('t', [1.0])})).collect())") &&
      uv venv -q --python 3.12 "$work/fast" &&
      uv pip install -q --python "$work/fast/bin/python" "$(ls "$work"/dist/*.whl)[fast]" &&
      (cd "$work" && "$work/fast/bin/python" -c \
        "import numpy as np, tidy3a; from labelarray import sums; v = np.ones(200_000, np.float32); assert sums._compiled('grouped_estimate') is not None, 'compiled kernel not loaded'; print(sums.reduce(v, 'mean', (0,), codes=np.zeros(200_000, np.intp), groups=1))")
  }
  run "source distribution, metadata, and installed wheel (plain, with the xarray extra, and with the fast extra)" distribution_job
fi

if (( parallel )); then
  wait
  for i in "${!names[@]}"; do
    printf '\n── %s ──\n' "${names[i]}"
    cat "${logs[i]}"
    if [[ "$(cat "${logs[i]}.status")" == 0 ]]; then
      printf '   ok: %s\n' "${names[i]}"
    else
      printf '   FAILED: %s\n' "${names[i]}"
      failed+=("${names[i]}")
    fi
  done
fi

printf '\n'
if [[ ${#failed[@]} -gt 0 ]]; then
  printf 'FAILED: %s\n' "${failed[@]}"
  exit 1
fi
printf 'All checks passed; safe to push.\n'
