#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Export completed artifacts on success, scientific stop, or a failed stage.
# The runner downloads only this selected bundle, then destroys its instance.
collect_artifacts() {
    local workload_exit=$?
    trap - EXIT
    .venv/bin/python scripts/collect_lam_v5_artifacts.py || exit 74
    exit "$workload_exit"
}
trap collect_artifacts EXIT
# A finite workload timeout also bounds negative or stalled experiments.
timeout --signal=TERM --kill-after=60 10800 .venv/bin/python scripts/lam_v5_campaign.py
