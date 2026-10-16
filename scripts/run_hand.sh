#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -f "$PROJECT_ROOT/.env.local" ]]; then source "$PROJECT_ROOT/.env.local"; fi
: "${ISAACSIM_ROOT:=$HOME/isaacsim}"
: "${CUROBO_ROOT:=$HOME/curobo}"
: "${ABI0_ROOT:=$HOME/depth-camera-check/abi0-recovery}"
: "${CUDA_HOME:=/usr/local/cuda-11.8}"
for required in "$ISAACSIM_ROOT/python.sh" "$CUROBO_ROOT/src/curobo/__init__.py" "$ABI0_ROOT/install/lib/libnvblox_lib.so" "$ABI0_ROOT/nvblox_torch/src/nvblox_torch/bin/libpy_nvblox.so"; do
  if [[ ! -f "$required" ]]; then echo "Missing verified dependency: $required. See docs/reproduction.md" >&2; exit 1; fi
done
export CUDA_HOME CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11
export FRANKA_DEMO_ROOT="$PROJECT_ROOT/experiment"
export PYTHONPATH="$FRANKA_DEMO_ROOT:$ABI0_ROOT/nvblox_torch/src:$CUROBO_ROOT/src"
export LD_LIBRARY_PATH="$ABI0_ROOT/install/lib:$CUDA_HOME/lib64"
export MPLCONFIGDIR="$FRANKA_DEMO_ROOT/mpl-cache" TORCH_EXTENSIONS_DIR="$ABI0_ROOT/torch-cache" PYTHONDONTWRITEBYTECODE=1
mkdir -p "$FRANKA_DEMO_ROOT"/{results,inputs,assets,mpl-cache}
cd "$FRANKA_DEMO_ROOT"
exec "$ISAACSIM_ROOT/python.sh" "$FRANKA_DEMO_ROOT/hand_experiment.py" "$@"
