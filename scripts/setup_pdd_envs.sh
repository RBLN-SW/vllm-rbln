#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="${PDD_VLLM_RBLN_REPO:-$(cd -- "${script_dir}/.." && pwd)}"
venv_root="${PDD_VENV_ROOT:-${repo_dir}/.venvs}"
prefill_venv="${PDD_PREFILL_VENV:-${venv_root}/pdd-prefill}"
decode_venv="${PDD_DECODE_VENV:-${venv_root}/pdd-decode}"
python_bin="${PDD_BOOTSTRAP_PYTHON:-python3}"

if ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "[setup:error] Python executable not found: ${python_bin}" >&2
    exit 1
fi

python_version="$("${python_bin}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
case "${python_version}" in
    3.10|3.11|3.12|3.13) ;;
    *)
        echo "[setup:error] Python 3.10-3.13 is required; found ${python_version}." >&2
        exit 1
        ;;
esac

create_venv() {
    local target="$1"
    if [[ ! -x "${target}/bin/python3" ]]; then
        echo "[setup] Creating ${target}"
        "${python_bin}" -m venv "${target}"
    fi
    "${target}/bin/python3" -m pip install --upgrade "pip>=24" setuptools wheel
}

create_venv "${prefill_venv}"
echo "[setup] Installing CUDA prefill dependencies"
"${prefill_venv}/bin/python3" -m pip install \
    -r "${repo_dir}/requirements/pdd-prefill.txt"

create_venv "${decode_venv}"
echo "[setup] Installing RBLN decode dependencies and this checkout"
"${decode_venv}/bin/python3" -m pip install \
    -r "${repo_dir}/requirements/pdd-decode.txt" \
    --extra-index-url https://wheels.vllm.ai/0.18.0/cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu
"${decode_venv}/bin/python3" -m pip install \
    --editable "${repo_dir}" \
    --extra-index-url https://wheels.vllm.ai/0.18.0/cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

echo "[setup] Environments are ready:"
echo "[setup]   prefill: ${prefill_venv}"
echo "[setup]   decode:  ${decode_venv}"
