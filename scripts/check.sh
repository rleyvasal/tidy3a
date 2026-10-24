#!/usr/bin/env bash
# Run what CI runs (.github/workflows/tests.yml), locally, before pushing.
#
# Each job gets a fresh environment with only tidy3A's declared
# dependencies, so nothing installed on this machine (tidy3, plot3, a newer
# xarray) can hide a failure that CI would see.
#
#   scripts/check.sh            all jobs
#   scripts/check.sh 3.12       one Python version only
#
# Needs uv (https://docs.astral.sh/uv/); it downloads missing Pythons.
set -euo pipefail

cd "$(dirname "$0")/.."
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

versions=("${@:-3.11 3.12 3.13 3.14}")
versions=(${versions[@]})
failed=()

run() {
  local name="$1"; shift
  printf '\n── %s ──\n' "$name"
  if "$@"; then
    printf '   ok: %s\n' "$name"
  else
    printf '   FAILED: %s\n' "$name"
    failed+=("$name")
  fi
}

pytest_job() {
  local python="$1"; shift
  local env="$work/py$python-$RANDOM"
  uv venv -q --python "$python" "$env" &&
    uv pip install -q --python "$env/bin/python" "$@" &&
    "$env/bin/python" -m pytest -q -p no:cacheprovider
}

for python in "${versions[@]}"; do
  run "pytest (Python $python)" pytest_job "$python" -e ".[dev]"
done

if [[ $# -eq 0 ]]; then
  run "pytest (minimum NumPy and xarray)" \
    pytest_job 3.11 "numpy==2.0.*" "pandas==2.2.2" "xarray==2024.6.0" "dask[array]" pytest -e .

  distribution_job() {
    uv build -q --out-dir "$work/dist" &&
      uvx twine check --strict "$work"/dist/* &&
      uv venv -q --python 3.12 "$work/clean" &&
      uv pip install -q --python "$work/clean/bin/python" "$work"/dist/*.whl &&
      (cd "$work" && "$work/clean/bin/python" -c \
        "import tidy3a, numpy as np; print(tidy3a.read_array(np.zeros(3), dims=['t']))")
  }
  run "source distribution, metadata, and installed wheel" distribution_job
fi

printf '\n'
if [[ ${#failed[@]} -gt 0 ]]; then
  printf 'FAILED: %s\n' "${failed[@]}"
  exit 1
fi
printf 'All checks passed; safe to push.\n'
