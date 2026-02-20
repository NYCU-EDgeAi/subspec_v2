#!/usr/bin/env bash
set -euo pipefail

# ---------- env ----------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate /home/scott306lr/envs/subspec
fi
export PYTHONPATH="$PWD"

# ---------- knobs ----------
SAMPLES="${SAMPLES:-50}"
MAX_LEN="${MAX_LEN:-4096}"
SEED="${SEED:-0}"
SHUFFLE="${SHUFFLE:-1}"
TOKEN_KL="${TOKEN_KL:-0}"                    # requested: 0/1
LANE="${LANE:-behavior}"                     # behavior or distribution
BENCHMARKS="${BENCHMARKS:-${BEHAVIOR_BENCHES:-gsm8k,human-eval}}"

FP16_CFG="${FP16_CFG:-configs/methods/vanilla.yaml}"
# Comma-separated list in "name=config_path" format.
COMPARE_METHODS="${COMPARE_METHODS:-int4=configs/methods/vanilla_int4.yaml,lossless_sd=configs/methods/subspec_sd_no_offload.yaml,lossy_sd=configs/methods/subspec_sd_lossy_no_offload.yaml}"

TMP_CFG_DIR="${TMP_CFG_DIR:-/tmp/subspec_bench_cfg}"
mkdir -p "$TMP_CFG_DIR"

# ---------- helper ----------
latest_run_dir() {
  ls -1dt experiments/*/"$1" | head -n1
}

prepare_cfg() {
  local src="$1"
  local dst="$2"
  cp "$src" "$dst"
  if grep -q '^compile_mode:' "$dst"; then
    sed -i 's/^compile_mode:.*/compile_mode: null/' "$dst"
  else
    echo "compile_mode: null" >> "$dst"
  fi
  if grep -q '^warmup_iter:' "$dst"; then
    sed -i 's/^warmup_iter:.*/warmup_iter: 0/' "$dst"
  else
    echo "warmup_iter: 0" >> "$dst"
  fi
}

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

if [[ "$SHUFFLE" == "1" ]]; then
  SHUFFLE_FLAG=(--shuffle)
else
  SHUFFLE_FLAG=()
fi

if [[ "$LANE" != "behavior" && "$LANE" != "distribution" ]]; then
  echo "Invalid LANE='$LANE'. Supported: behavior, distribution" >&2
  exit 1
fi

EFFECTIVE_TOKEN_KL="$TOKEN_KL"
if [[ "$LANE" == "behavior" && "$TOKEN_KL" == "1" ]]; then
  echo "Note: token_kl is disabled in behavior lane; forcing TOKEN_KL=0."
  EFFECTIVE_TOKEN_KL="0"
fi

if [[ "$EFFECTIVE_TOKEN_KL" == "1" ]]; then
  TOKEN_KL_FLAG=()
else
  TOKEN_KL_FLAG=(--no-token-kl)
fi

IFS=',' read -ra RAW_METHOD_ITEMS <<< "$COMPARE_METHODS"
COMPARE_NAMES=()
COMPARE_CFGS=()
for item in "${RAW_METHOD_ITEMS[@]}"; do
  item="$(trim "$item")"
  [[ -z "$item" ]] && continue
  if [[ "$item" != *=* ]]; then
    echo "Invalid COMPARE_METHODS entry: '$item' (expected name=config_path)" >&2
    exit 1
  fi
  name="$(trim "${item%%=*}")"
  cfg="$(trim "${item#*=}")"
  if [[ -z "$name" || -z "$cfg" ]]; then
    echo "Invalid COMPARE_METHODS entry: '$item' (empty name or config)" >&2
    exit 1
  fi
  COMPARE_NAMES+=("$name")
  COMPARE_CFGS+=("$cfg")
done

if [[ ${#COMPARE_NAMES[@]} -eq 0 ]]; then
  echo "No compare methods configured. Set COMPARE_METHODS." >&2
  exit 1
fi

# ---------- temp configs ----------
TMP_FP16_CFG="$TMP_CFG_DIR/vanilla.yaml"
TMP_COMPARE_CFGS=()

if [[ ! -f "$FP16_CFG" ]]; then
  echo "Base config not found: $FP16_CFG" >&2
  exit 1
fi
prepare_cfg "$FP16_CFG" "$TMP_FP16_CFG"
for i in "${!COMPARE_CFGS[@]}"; do
  cfg="${COMPARE_CFGS[$i]}"
  if [[ ! -f "$cfg" ]]; then
    echo "Compare config not found for '${COMPARE_NAMES[$i]}': $cfg" >&2
    exit 1
  fi
  tmp_cfg="$TMP_CFG_DIR/compare_${i}_$(basename "$cfg")"
  prepare_cfg "$cfg" "$tmp_cfg"
  TMP_COMPARE_CFGS+=("$tmp_cfg")
done

echo "Resolved configs:"
echo "  fp16: $TMP_FP16_CFG"
for i in "${!COMPARE_NAMES[@]}"; do
  echo "  ${COMPARE_NAMES[$i]}: ${TMP_COMPARE_CFGS[$i]}"
done

echo ""
echo "Planned comparison tasks:"
echo "  lane      : $LANE"
echo "  benchmarks: $BENCHMARKS"
echo "  samples   : $SAMPLES"
echo "  max_len   : $MAX_LEN"
echo "  shuffle   : $([[ "$SHUFFLE" == "1" ]] && echo on || echo off)"
echo "  token_kl  : $([[ "$EFFECTIVE_TOKEN_KL" == "1" ]] && echo enabled || echo disabled)"
for i in "${!COMPARE_NAMES[@]}"; do
  idx=$((i + 1))
  echo "  ${idx}) fp16 vs ${COMPARE_NAMES[$i]}  (config: ${TMP_COMPARE_CFGS[$i]})"
done
echo ""

# ---------- paired compare (acc + flips in one run) ----------
BASELINE_DIR=""
COMPARE_LABELS=()
COMPARE_DIRS=()

for i in "${!COMPARE_NAMES[@]}"; do
  name="${COMPARE_NAMES[$i]}"
  cfg="${TMP_COMPARE_CFGS[$i]}"

  echo ""
  phase=$((i + 1))
  total="${#COMPARE_NAMES[@]}"
  mode="build baseline"
  EXTRA_ARGS=()
  if [[ -n "$BASELINE_DIR" ]]; then
    mode="reuse baseline"
    EXTRA_ARGS=(--reuse-baseline-dir "$BASELINE_DIR")
  fi
  echo "[$phase/$total] Running compare: fp16 vs ${name} ($mode)"

  python -m run.main \
    --config "$TMP_FP16_CFG" \
    --max-length "$MAX_LEN" \
    run-benchmark-compare \
    --compare-config "$cfg" \
    --compare-name "$name" \
    --benchmarks "$BENCHMARKS" \
    --lane "$LANE" \
    --max-samples "$SAMPLES" \
    --seed "$SEED" \
    "${SHUFFLE_FLAG[@]}" \
    "${TOKEN_KL_FLAG[@]}" \
    "${EXTRA_ARGS[@]}"

  run_dir="$(latest_run_dir run_benchmark_compare)"
  if [[ -z "$BASELINE_DIR" ]]; then
    BASELINE_DIR="$run_dir"
  fi
  COMPARE_LABELS+=("fp16_vs_${name}")
  COMPARE_DIRS+=("$run_dir")
done

# ---------- compact summary ----------
COMPARE_LABELS_JOINED="$(IFS='|'; echo "${COMPARE_LABELS[*]}")"
COMPARE_DIRS_JOINED="$(IFS='|'; echo "${COMPARE_DIRS[*]}")"
COMPARE_LABELS="$COMPARE_LABELS_JOINED" \
COMPARE_DIRS="$COMPARE_DIRS_JOINED" \
BENCHES="$BENCHMARKS" \
python - <<'PY'
import json
import os
import pathlib

def load_json_objects(path):
    text = pathlib.Path(path).read_text().strip()
    dec = json.JSONDecoder()
    out = []
    idx = 0
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        obj, idx = dec.raw_decode(text, idx)
        out.append(obj)
    return out

def read_cmp(run_dir, bench):
    rec = load_json_objects(pathlib.Path(run_dir) / bench / "results.jsonl")[-1][bench]
    return {
        "base_acc": rec.get("base_accuracy"),
        "cmp_acc": rec.get("compare_accuracy"),
        "flips_rate": rec.get("flips_rate"),
        "flips_c2i_count": rec.get("flips_c2i_count"),
        "flips_i2c_count": rec.get("flips_i2c_count"),
        "allflips_rate": rec.get("allflips_rate"),
        "output_change_rate": rec.get("output_change_rate"),
        "wrong_to_wrong_change_rate": rec.get("wrong_to_wrong_change_rate"),
    }

def fmtf(v):
    if isinstance(v, (int, float)):
        return f"{float(v):.3f}"
    return "N/A"

benches = [b.strip() for b in os.environ["BENCHES"].split(",") if b.strip()]
labels = [x for x in os.environ["COMPARE_LABELS"].split("|") if x]
dirs = [x for x in os.environ["COMPARE_DIRS"].split("|") if x]
if len(labels) != len(dirs):
    raise SystemExit("compare labels/dirs length mismatch")

print("\n=== COMPARE (acc + flips) ===")
for name, run_dir in zip(labels, dirs):
    print(f"[{name}] {run_dir}")
    for bench in benches:
        r = read_cmp(run_dir, bench)
        print(
            f"  {bench:10s} base={fmtf(r['base_acc'])} cmp={fmtf(r['cmp_acc'])} "
            f"flips_rate={fmtf(r['flips_rate'])} c2i={r['flips_c2i_count']} i2c={r['flips_i2c_count']} "
            f"allflips_rate={r['allflips_rate']} output_change_rate={r['output_change_rate']} "
            f"wrong_to_wrong_change_rate={r['wrong_to_wrong_change_rate']}"
        )
    print()
PY

echo "Done."
echo "COMPARE dirs:"
for i in "${!COMPARE_LABELS[@]}"; do
  echo "  ${COMPARE_LABELS[$i]} -> ${COMPARE_DIRS[$i]}"
done
