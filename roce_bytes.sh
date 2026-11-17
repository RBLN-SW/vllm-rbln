#!/usr/bin/env bash
# Snapshot RoCE TX/RX bytes per RDMA device.
#   ./roce_bytes.sh > before.txt         # snapshot
#   ... run the transfer ...
#   ./roce_bytes.sh before.txt           # delta since that snapshot
set -euo pipefail

snapshot() {
  for dev in /sys/class/infiniband/*; do
    [ -e "$dev" ] || continue
    for port in "$dev"/ports/*; do
      name="$(basename "$dev")/$(basename "$port")"
      netdev=$(ls "$dev/device/net" 2>/dev/null | head -1)
      state=$(awk '{print $2}' "$port/state")
      if [ -r "$port/hw_counters/tx_bytes" ]; then
        tx=$(cat "$port/hw_counters/tx_bytes")
        rx=$(cat "$port/hw_counters/rx_bytes")
      else
        # IB port counters count 4-byte words.
        tx=$(( $(cat "$port/counters/port_xmit_data") * 4 ))
        rx=$(( $(cat "$port/counters/port_rcv_data") * 4 ))
      fi
      echo "$name ${netdev:--} $state $tx $rx"
    done
  done
}

human() { numfmt --to=iec-i --suffix=B "$1"; }

if [ $# -eq 0 ]; then
  snapshot
  exit 0
fi

before=$1
printf "%-20s %-16s %-7s %14s %14s\n" DEVICE NETDEV STATE TX_DELTA RX_DELTA
snapshot | while read -r name netdev state tx rx; do
  read -r _ _ _ tx0 rx0 < <(grep "^$name " "$before" || echo "- - - $tx $rx")
  printf "%-20s %-16s %-7s %14s %14s\n" "$name" "$netdev" "$state" \
    "$(human $((tx - tx0)))" "$(human $((rx - rx0)))"
done
