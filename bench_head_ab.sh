#!/usr/bin/env bash
# A/B benchmark: candle DPT head vs candle-free fast_conv head.
#
# Both runs use the fast backbone (DA_FAST_ATTN=1) so the comparison isolates
# the head difference. Interleaved with cooldowns to control for thermal noise.
#
# Usage:
#   ./bench_head_ab.sh [REPEAT] [MODEL] [INPUT] [COOLDOWN_SEC]
#
# Defaults: REPEAT=5, MODEL=q5_k, INPUT=canyon.jpg, COOLDOWN=15s

set -euo pipefail

REPEAT="${1:-5}"
MODEL="${2:-models/depth-anything-base-q5_k.gguf}"
INPUT="${3:-assets/samples/canyon.jpg}"
COOLDOWN="${4:-15}"

echo "[bench_head_ab] repeat=$REPEAT model=$MODEL input=$INPUT cooldown=${COOLDOWN}s"

TMP=$(mktemp -d)
CANDLE_JSON="$TMP/candle.txt"
FAST_JSON="$TMP/fast.txt"
trap 'rm -rf "$TMP"' EXIT

run_one() {
    local label="$1"
    local head_flag="$2"
    local out="$3"
    echo "[bench_head_ab] running $label (cooldown ${COOLDOWN}s after)..."
    # Backbone always fast (DA_FAST_ATTN=1); head toggled by DA_FAST_HEAD.
    DA_FAST_ATTN=1 DA_FAST_HEAD="$head_flag" RAYON_NUM_THREADS=16 \
        ./target/release/examples/bench \
        --model "$MODEL" --input "$INPUT" --warmup 3 --repeat 1 2>/dev/null \
        | tee "$out" >/dev/null
    local bb=$(grep -oP '"backbone_ms":\K[0-9.]+' "$out" | head -1)
    local head=$(grep -oP '"head_ms":\K[0-9.]+' "$out" | head -1)
    local infer=$(grep -oP '"infer_mean_ms":\K[0-9.]+' "$out" | head -1)
    echo "  $label: infer=${infer}ms backbone=${bb}ms head=${head}ms"
    echo "$infer $bb $head" >> "${out}.nums"
    sleep "$COOLDOWN"
}

echo ""
echo "==== Interleaved head A/B (candle head vs fast_conv head) ===="
echo "Both runs use DA_FAST_ATTN=1 (fast backbone)."
echo ""

for i in $(seq 1 "$REPEAT"); do
    echo "--- pair $i/$REPEAT ---"
    run_one "candle" "0" "$CANDLE_JSON"
    run_one "fast  " "1" "$FAST_JSON"
done

echo ""
echo "==== Summary ===="

median() {
    sort -n | awk '{a[NR]=$1} END{if(NR%2==1) print a[(NR+1)/2]; else print (a[NR/2]+a[NR/2+1])/2}'
}

candle_infer=$(cut -d' ' -f1 "${CANDLE_JSON}.nums" | median)
candle_bb=$(cut -d' ' -f2 "${CANDLE_JSON}.nums" | median)
candle_head=$(cut -d' ' -f3 "${CANDLE_JSON}.nums" | median)
fast_infer=$(cut -d' ' -f1 "${FAST_JSON}.nums" | median)
fast_bb=$(cut -d' ' -f2 "${FAST_JSON}.nums" | median)
fast_head=$(cut -d' ' -f3 "${FAST_JSON}.nums" | median)

echo "                   candle (median)    fast (median)    speedup"
printf "  infer_total  :  %8.1f ms      %8.1f ms      %.2fx\n" \
    "$candle_infer" "$fast_infer" "$(awk "BEGIN{printf \"%.3f\", $candle_infer / $fast_infer}")"
printf "  backbone     :  %8.1f ms      %8.1f ms      %.2fx\n" \
    "$candle_bb" "$fast_bb" "$(awk "BEGIN{printf \"%.3f\", $candle_bb / $fast_bb}")"
printf "  head         :  %8.1f ms      %8.1f ms      %.2fx\n" \
    "$candle_head" "$fast_head" "$(awk "BEGIN{printf \"%.3f\", $candle_head / $fast_head}")"

echo ""
echo "Raw candle runs: $(cat "${CANDLE_JSON}.nums" | tr '\n' '; ')"
echo "Raw fast runs:   $(cat "${FAST_JSON}.nums" | tr '\n' '; ')"
