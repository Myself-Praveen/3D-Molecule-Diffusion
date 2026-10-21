#!/usr/bin/env bash
# Conda-free deployment entry point.
#
# Expects deploy/deploy-env to already exist (built by deploy/deploy.sh build
# or deploy/deploy.sh run). This script is what ships alongside an artifact and
# what a user actually invokes.
set -euo pipefail
dir="$(cd "$(dirname "$0")/.." && pwd)"
venv="$dir/deploy/deploy-env"

if [ ! -f "$venv/bin/python" ]; then
    echo "[deploy] environment not found at $venv — rebuild with: bash deploy/deploy.sh build" >&2
    exit 2
fi

export VIRTUAL_ENV="$venv"
export PATH="$venv/bin:$PATH"
export PYTHONNOUSERSITE=1

echo "[deploy] using environment $venv"
"$venv/bin/python" - <<'PY'
import sys, platform
try:
    import torch, numpy, scipy, matplotlib, pandas
    print(f"[deploy] platform={platform.platform()}")
    print(f"[deploy] python={sys.version.splitlines()[0]}")
    print(f"[deploy] torch={torch.__version__} numpy={numpy.__version__} scipy={scipy.__version__} matplotlib={matplotlib.__version__} pandas={pandas.__version__}")
except Exception as e:
    print(f"[deploy] import failure: {e}", file=sys.stderr)
    sys.exit(2)
PY

# Dispatch to the real entry point. Replace the line below with whatever the
# artifact's primary command is. Keep it a single Python invocation so the env
# setup above is the only deployment concern.
exec "$venv/bin/python" - <<'PY'
print("[deploy] entry point placeholder — replace deploy/run.sh body with the real command")
PY
