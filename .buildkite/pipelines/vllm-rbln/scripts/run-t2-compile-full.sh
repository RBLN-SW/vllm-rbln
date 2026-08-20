#!/usr/bin/env bash
# T1 at full depth (0 = every layer). After merge, not on PRs.

export NUM_HIDDEN_LAYERS=0
# exec keeps t1's $0, so the lane name has to come from here.
export LANE_NAME=t2-compile-full
# --maxfail=0 lands after t1's -x and overrides it: at full depth every model's
# result is wanted, not just the first failure.
exec "$(dirname "${BASH_SOURCE[0]}")/run-t1-compile.sh" --maxfail=0 "$@"
