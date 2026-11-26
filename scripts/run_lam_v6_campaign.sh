#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export FLUIDGENIE_LAM_START_MARKER="delivery/lam_v6/training_started.json"
collect_artifacts() {
    local workload_exit=$?
    trap - EXIT
    .venv/bin/python scripts/collect_lam_v5_artifacts.py --campaign lam_v6 --reuse-data || exit 74
    exit "$workload_exit"
}
trap collect_artifacts EXIT
timeout --signal=TERM --kill-after=60 10800 .venv/bin/python scripts/lam_v6_campaign.py
