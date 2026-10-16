#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
condition="${1:-R}"
case "$condition" in R|LOW|HIGH) ;; *) echo 'Usage: scripts/replay.sh R|LOW|HIGH [extra display arguments]' >&2; exit 2;; esac
if [[ $# -gt 0 ]]; then shift; fi
# HIGH in the featured result is p=1 (73.515982 mm), not default p=0.5.
policy=()
if [[ "$condition" == HIGH ]]; then policy=(--eta-exponent 1.0); fi
exec "$PROJECT_ROOT/scripts/run_hand.sh" \
  --mode replay --condition "$condition" "${policy[@]}" \
  --input "$PROJECT_ROOT/data/hand-tape-20261012-000004-472steps-registered" \
  --goal-mode reached --goal-plan AB --state-input measured \
  --model-dt 0.016666666666666666 --failure-action hold-measured \
  --audit-failures --headless --light-render "$@"
