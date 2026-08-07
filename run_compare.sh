#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate /home/scott306lr/envs/subspec
fi
export PYTHONPATH="$PWD"

# ---------- knobs ----------
SAMPLES="${SAMPLES:-200}"
MAX_LEN="${MAX_LEN:-4096}"
SEED="${SEED:-0}"
SHUFFLE="${SHUFFLE:-1}"
TOKEN_KL="${TOKEN_KL:-0}"                      # requested: 0/1
LANE="${LANE:-behavior}"                       # behavior or distribution
BENCHMARKS="${BENCHMARKS:-${BEHAVIOR_BENCHES:-gsm8k,human-eval-instruct}}"

FP16_CFG="${FP16_CFG:-configs/methods/vanilla.yaml}"
COMPARE_METHODS="${COMPARE_METHODS:-}"          # newline- or comma-separated "name=config_path"
COMPARE_METHODS_FILE="${COMPARE_METHODS_FILE:-}" # one "name=config_path" per line
DEFAULT_COMPARE_METHODS=$(
  cat <<'EOF'
int4=configs/methods/vanilla_int4.yaml
kv_rewrite_32=configs/methods/kv_rewrite_no_offload_32.yaml
kv_rewrite_128=configs/methods/kv_rewrite_no_offload_128.yaml
kv_rewrite_256=configs/methods/kv_rewrite_no_offload_256.yaml
lossy_sd=configs/methods/subspec_sd_lossy_no_offload.yaml
lossless_sd=configs/methods/subspec_sd_no_offload.yaml
EOF
)

# Resume/reuse controls.
# REUSE_BASELINE_DIR must point to a previous run's "run_benchmark_compare" directory.
REUSE_BASELINE_DIR="${REUSE_BASELINE_DIR:-${BASELINE_DIR:-}}"
START_FROM="${START_FROM:-1}"                  # 1-based method index

# Optional CLI overrides. Defaults respect YAML.
WARMUP_ITER="${WARMUP_ITER:-}"                 # e.g. 0
DISABLE_COMPILE="${DISABLE_COMPILE:-0}"        # 1 => force --compile-mode none

# Required for HumanEval (`evaluate` code_eval executes untrusted code).
if [[ ",$BENCHMARKS," == *",human-eval,"* || ",$BENCHMARKS," == *",human-eval-instruct,"* ]]; then
  export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"
fi

latest_run_dir() {
  ls -1dt experiments/*/"$1" 2>/dev/null | head -n1
}

trim() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

parse_method_specs() {
  local raw="$1"
  local specs=()
  local line=""

  while IFS= read -r line; do
    line="${line%%#*}" # strip inline comments
    line="$(trim "$line")"
    [[ -z "$line" ]] && continue
    specs+=("$line")
  done <<< "$(printf '%s\n' "$raw" | tr ',' '\n')"

  if [[ ${#specs[@]} -eq 0 ]]; then
    echo "No compare methods configured." >&2
    echo "Provide COMPARE_METHODS, COMPARE_METHODS_FILE, or edit DEFAULT_COMPARE_METHODS." >&2
    return 1
  fi

  printf '%s\n' "${specs[@]}"
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

if ! [[ "$START_FROM" =~ ^[0-9]+$ ]] || [[ "$START_FROM" -lt 1 ]]; then
  echo "Invalid START_FROM='$START_FROM'. Must be an integer >= 1." >&2
  exit 1
fi

if [[ ! -f "$FP16_CFG" ]]; then
  echo "Base config not found: $FP16_CFG" >&2
  exit 1
fi

if [[ -n "$REUSE_BASELINE_DIR" && ! -d "$REUSE_BASELINE_DIR" ]]; then
  echo "REUSE_BASELINE_DIR not found: $REUSE_BASELINE_DIR" >&2
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

MODEL_OVERRIDES=(--set "max_length=$MAX_LEN")
if [[ -n "$WARMUP_ITER" ]]; then
  MODEL_OVERRIDES+=(--set "warmup_iter=$WARMUP_ITER")
fi
if [[ "$DISABLE_COMPILE" == "1" ]]; then
  MODEL_OVERRIDES+=(--set "compile_mode=none")
fi

RAW_METHOD_ITEMS=()
if [[ -n "$COMPARE_METHODS_FILE" ]]; then
  if [[ ! -f "$COMPARE_METHODS_FILE" ]]; then
    echo "COMPARE_METHODS_FILE not found: $COMPARE_METHODS_FILE" >&2
    exit 1
  fi
  while IFS= read -r line; do
    RAW_METHOD_ITEMS+=("$line")
  done < <(parse_method_specs "$(cat "$COMPARE_METHODS_FILE")")
elif [[ -n "$COMPARE_METHODS" ]]; then
  while IFS= read -r line; do
    RAW_METHOD_ITEMS+=("$line")
  done < <(parse_method_specs "$COMPARE_METHODS")
else
  while IFS= read -r line; do
    RAW_METHOD_ITEMS+=("$line")
  done < <(parse_method_specs "$DEFAULT_COMPARE_METHODS")
fi

COMPARE_NAMES=()
COMPARE_CFGS=()
for item in "${RAW_METHOD_ITEMS[@]}"; do
  if [[ "$item" != *=* ]]; then
    echo "Invalid compare method entry: '$item' (expected name=config_path)" >&2
    exit 1
  fi

  name="$(trim "${item%%=*}")"
  cfg="$(trim "${item#*=}")"
  if [[ -z "$name" || -z "$cfg" ]]; then
    echo "Invalid compare method entry: '$item' (empty name or config)" >&2
    exit 1
  fi
  if [[ ! -f "$cfg" ]]; then
    echo "Compare config not found for '$name': $cfg" >&2
    exit 1
  fi

  COMPARE_NAMES+=("$name")
  COMPARE_CFGS+=("$cfg")
done

if [[ ${#COMPARE_NAMES[@]} -eq 0 ]]; then
  echo "No compare methods configured." >&2
  exit 1
fi

if [[ "$START_FROM" -gt "${#COMPARE_NAMES[@]}" ]]; then
  echo "START_FROM=$START_FROM exceeds method count (${#COMPARE_NAMES[@]})." >&2
  exit 1
fi

echo "Planned comparison tasks:"
echo "  fp16 config : $FP16_CFG"
if [[ -n "$COMPARE_METHODS_FILE" ]]; then
  echo "  methods src : file $COMPARE_METHODS_FILE"
elif [[ -n "$COMPARE_METHODS" ]]; then
  echo "  methods src : COMPARE_METHODS env"
else
  echo "  methods src : script default block"
fi
echo "  lane        : $LANE"
echo "  benchmarks  : $BENCHMARKS"
echo "  samples     : $SAMPLES"
echo "  max_len     : $MAX_LEN"
echo "  shuffle     : $([[ "$SHUFFLE" == "1" ]] && echo on || echo off)"
echo "  token_kl    : $([[ "$EFFECTIVE_TOKEN_KL" == "1" ]] && echo enabled || echo disabled)"
echo "  warmup_iter : $([[ -n "$WARMUP_ITER" ]] && echo "$WARMUP_ITER (override)" || echo "yaml/default")"
echo "  compile     : $([[ "$DISABLE_COMPILE" == "1" ]] && echo "disabled (override)" || echo "yaml/default")"
echo "  start_from  : $START_FROM"
if [[ -n "$REUSE_BASELINE_DIR" ]]; then
  echo "  baseline    : reuse $REUSE_BASELINE_DIR"
else
  echo "  baseline    : build from first executed method"
fi
if [[ ",$BENCHMARKS," == *",human-eval,"* || ",$BENCHMARKS," == *",human-eval-instruct,"* ]]; then
  echo "  code_eval   : HF_ALLOW_CODE_EVAL=${HF_ALLOW_CODE_EVAL:-unset}"
fi
for i in "${!COMPARE_NAMES[@]}"; do
  idx=$((i + 1))
  echo "  ${idx}) fp16 vs ${COMPARE_NAMES[$i]}  (config: ${COMPARE_CFGS[$i]})"
done
echo ""

BASELINE_DIR_RUNTIME="$REUSE_BASELINE_DIR"
RUN_LABELS=()
RUN_DIRS=()

for i in "${!COMPARE_NAMES[@]}"; do
  idx=$((i + 1))
  if [[ "$idx" -lt "$START_FROM" ]]; then
    echo "[$idx/${#COMPARE_NAMES[@]}] Skip fp16 vs ${COMPARE_NAMES[$i]} (before START_FROM=$START_FROM)"
    continue
  fi

  name="${COMPARE_NAMES[$i]}"
  cfg="${COMPARE_CFGS[$i]}"
  extra_args=()
  mode="build baseline"
  if [[ -n "$BASELINE_DIR_RUNTIME" ]]; then
    mode="reuse baseline"
    extra_args=(--reuse-baseline-dir "$BASELINE_DIR_RUNTIME")
  fi

  echo ""
  echo "[$idx/${#COMPARE_NAMES[@]}] Running compare: fp16 vs $name ($mode)"
  python -m run.main \
    --config "$FP16_CFG" \
    "${MODEL_OVERRIDES[@]}" \
    run-benchmark-compare \
    --compare-config "$cfg" \
    --compare-name "$name" \
    --benchmarks "$BENCHMARKS" \
    --lane "$LANE" \
    --max-samples "$SAMPLES" \
    --seed "$SEED" \
    "${SHUFFLE_FLAG[@]}" \
    "${TOKEN_KL_FLAG[@]}" \
    "${extra_args[@]}"

  run_dir="$(latest_run_dir run_benchmark_compare)"
  if [[ -z "$run_dir" ]]; then
    echo "Failed to locate run_benchmark_compare output directory." >&2
    exit 1
  fi

  if [[ -z "$BASELINE_DIR_RUNTIME" ]]; then
    BASELINE_DIR_RUNTIME="$run_dir"
  fi

  RUN_LABELS+=("fp16_vs_${name}")
  RUN_DIRS+=("$run_dir")
done

if [[ ${#RUN_LABELS[@]} -eq 0 ]]; then
  echo "No comparisons executed."
  exit 0
fi

RUN_LABELS_JOINED="$(IFS='|'; echo "${RUN_LABELS[*]}")"
RUN_DIRS_JOINED="$(IFS='|'; echo "${RUN_DIRS[*]}")"
RUN_LABELS_ENV="$RUN_LABELS_JOINED" \
RUN_DIRS_ENV="$RUN_DIRS_JOINED" \
BENCHES_ENV="$BENCHMARKS" \
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


benches = [b.strip() for b in os.environ["BENCHES_ENV"].split(",") if b.strip()]
labels = [x for x in os.environ["RUN_LABELS_ENV"].split("|") if x]
dirs = [x for x in os.environ["RUN_DIRS_ENV"].split("|") if x]
if len(labels) != len(dirs):
    raise SystemExit("label/dir length mismatch")

print("\n=== COMPARE (acc + flips) ===")
for name, run_dir in zip(labels, dirs):
    print(f"[{name}] {run_dir}")
    for bench in benches:
        r = read_cmp(run_dir, bench)
        line = (
            f"  {bench:20s} base={fmtf(r['base_acc'])} cmp={fmtf(r['cmp_acc'])} "
            f"flips_rate={fmtf(r['flips_rate'])} c2i={r['flips_c2i_count']} i2c={r['flips_i2c_count']} "
            f"allflips_rate={r['allflips_rate']} output_change_rate={r['output_change_rate']} "
            f"wrong_to_wrong_change_rate={r['wrong_to_wrong_change_rate']}"
        )
        print(line)
    print()
PY

echo "Done."
echo "Baseline dir used: $BASELINE_DIR_RUNTIME"
echo "COMPARE dirs:"
for i in "${!RUN_LABELS[@]}"; do
  echo "  ${RUN_LABELS[$i]} -> ${RUN_DIRS[$i]}"
done
