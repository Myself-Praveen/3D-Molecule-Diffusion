#!/usr/bin/env bash
# Deploy helper — conda-free, isolated environment at deploy/deploy-env.
#
# Usage:
#   bash deploy/deploy.sh run   # builds the env (once) then runs deploy/run.sh
#   bash deploy/deploy.sh build # only build/create the env
#
# The env lives under the repo so it can be excluded from version control but
# shipped alongside artifacts. It is rebuilt from a pinned requirements file.
set -euo pipefail
dir="$(cd "$(dirname "$0")/.." && pwd)"
venv="$dir/deploy/deploy-env"
python_exe="$venv/bin/python"

need_build=false
case "${1:-run}" in
    build) need_build=true ;;
    run)   need_build=false ;;
    *) echo "usage: bash deploy/deploy.sh [build|run]" >&2; exit 2 ;;
esac

if [ ! -f "$python_exe" ] || [ ! -f "$venv/bin/pip" ]; then
    need_build=true
fi

if $need_build; then
    echo "[deploy] creating isolated environment at $venv"
    python3 -m venv "$venv"
    # pip list may differ by platform index; the pinning is intentionally loose
    # on lower bounds and lets pip resolve the newest compatible set. If a
    # deployment needs a pinned hash, regenerate deploy/requirements-freeze.txt
    # from this venv once built.
    "$venv/bin/pip" install --upgrade pip -q
    "$venv/bin/pip" install \
        torch torchvision torchaudio \
        numpy scipy matplotlib pandas requests \
        -q || { echo "[deploy] pip install failed" >&2; exit 2; }
    echo "[deploy] environment ready at $venv"
fi

if [ "${1:-run}" = "run" ]; then
    if [ ! -f "$dir/deploy/run.sh" ]; then
        echo "[deploy] deploy/run.sh missing" >&2; exit 2
    fi
    echo "[deploy] running deploy/run.sh"
    exec bash "$dir/deploy/run.sh" "$@"
fi
