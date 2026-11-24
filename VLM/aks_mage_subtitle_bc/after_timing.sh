#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
exec 9>logs/queue.lock
flock -n 9 || { echo "Queue already active"; exit 1; }
timing=../aks_mage_timing
# Wait for an already running measurement to release its own checkpoint lock.
lock="$timing/results/2e596ff212820ce3/run.lock"
if [[ -e "$lock" ]]; then flock "$lock" true; fi
echo "Stage 1: run or resume the paired timing experiment"
bash "$timing/run.sh"
test -s "$timing/results/2e596ff212820ce3/summary.csv"
echo "Stage 2: full subtitle audit and paired B/C generation"
bash run.sh
