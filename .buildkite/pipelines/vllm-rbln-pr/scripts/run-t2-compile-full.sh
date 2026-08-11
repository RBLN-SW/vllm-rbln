#!/usr/bin/env bash
# T2 -- the T1 lane at full depth. After merge, not on PRs.

export NUM_HIDDEN_LAYERS=0
exec "$(dirname "${BASH_SOURCE[0]}")/run-t1-compile.sh" "$@"
