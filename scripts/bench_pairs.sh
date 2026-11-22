#!/bin/bash
# Paired benchmark on one machine: the prototype (examples/circuit_amplitudes) and artensor
# (scripts/bench_artensor.py) take turns on the same tree, precision, thread count and device,
# with the order swapped every round, so load changes hit both. Each run is a separate process
# under /usr/bin/time -v (maximum RSS, page faults, user and system time).
#
# Usage: scripts/bench_pairs.sh OUT_DIR ROUNDS DEVICE [REPEATS]
#   DEVICE: a CPU thread count (e.g. 16) or cuda (the first visible CUDA device)
#   REPEATS: runs per process (default 1; use 2 or more on CUDA to see warm runs)
# Environment: SE (this crate, built with --release --examples), ART (an artensor checkout),
#   PY (a python with torch and numpy), BIN (the example binary, default
#   $SE/target/release/examples/circuit_amplitudes), DTYPES (default "c128 c64").
set -u
OUT=$1
ROUNDS=$2
DEVICE=$3
REPEATS=${4:-1}
SE=${SE:-$(cd "$(dirname "$0")/.." && pwd)}
BIN=${BIN:-$SE/target/release/examples/circuit_amplitudes}
DATA=$SE/data/n30m14
DTYPES=${DTYPES:-c128 c64}
mkdir -p "$OUT"
if [ "$DEVICE" = cuda ]; then
    T=1
    ADEV=cuda
else
    T=$DEVICE
    ADEV=cpu
fi
export OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T RAYON_NUM_THREADS=$T

run_tenferro() {
    echo "$(date +%T) round $1 tenferro $2 load $(cut -d' ' -f1-3 /proc/loadavg)"
    /usr/bin/time -v "$BIN" "$DATA" "$DEVICE" "$2" "$REPEATS" > "$OUT/tenferro_$2_r$1.log" 2>&1
}

run_artensor() {
    local adt=complex128
    [ "$2" = c64 ] && adt=complex64
    echo "$(date +%T) round $1 artensor $2 load $(cut -d' ' -f1-3 /proc/loadavg)"
    /usr/bin/time -v "$PY" "$SE/scripts/bench_artensor.py" "$DATA" "$ART" "$adt" "$ADEV" "$T" "$REPEATS" \
        > "$OUT/artensor_$2_r$1.log" 2>&1
}

for r in $(seq 1 "$ROUNDS"); do
    for dt in $DTYPES; do
        if [ $((r % 2)) -eq 1 ]; then
            run_tenferro "$r" "$dt"
            run_artensor "$r" "$dt"
        else
            run_artensor "$r" "$dt"
            run_tenferro "$r" "$dt"
        fi
    done
done
echo "$(date +%T) done, load $(cut -d' ' -f1-3 /proc/loadavg)"
