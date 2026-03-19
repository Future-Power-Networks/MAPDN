#!/usr/bin/env bash

# Usage:
#   bash train_case33_masked_maac.sh <alias> <voltage_barrier_type> <save_path> <graph_dir>
# Example:
#   bash train_case33_masked_maac.sh exp1 l1 trial_masked graph_cache/case33

if [ $# -lt 4 ]; then
  echo "Usage: bash $0 <alias> <voltage_barrier_type> <save_path> <graph_dir>"
  exit 1
fi

ALIAS=$1
VOLTAGE_BARRIER_TYPE=$2
SAVE_PATH=$3
GRAPH_DIR=$4

if [ ! -d "$SAVE_PATH" ]; then
  mkdir -p "$SAVE_PATH"
fi

if [ ! -d "$GRAPH_DIR" ]; then
  echo "Error: graph_dir does not exist -> $GRAPH_DIR"
  exit 1
fi

# 1) Vanilla MAAC baseline (full attention)
export CUDA_VISIBLE_DEVICES=0
nohup python -u train_masked_maac.py \
  --alias "0" \
  --mode distributed \
  --scenario case141_3min_final \
  --voltage-barrier-type "$VOLTAGE_BARRIER_TYPE" \
  --alg-config args/alg_args/ablations/maac_masked_full.yaml \
  --save-path "$SAVE_PATH" \
  > "$SAVE_PATH/maac_full_141_${ALIAS}_${VOLTAGE_BARRIER_TYPE}.out" 2>&1 &

# 2) Mask-only
export CUDA_VISIBLE_DEVICES=0
nohup python -u train_masked_maac.py \
  --alias "1" \
  --mode distributed \
  --scenario case141_3min_final \
  --voltage-barrier-type "$VOLTAGE_BARRIER_TYPE" \
  --graph-dir "$GRAPH_DIR" \
  --alg-config args/alg_args/ablations/maac_masked_mask_only.yaml \
  --save-path "$SAVE_PATH" \
  > "$SAVE_PATH/maac_mask_only_141_${ALIAS}_${VOLTAGE_BARRIER_TYPE}.out" 2>&1 &

# 3) Formal main model: mask + static prior bias (scale = 0.5)
export CUDA_VISIBLE_DEVICES=0
nohup python -u train_masked_maac.py \
  --alias "2" \
  --mode distributed \
  --scenario case141_3min_final \
  --voltage-barrier-type "$VOLTAGE_BARRIER_TYPE" \
  --graph-dir "$GRAPH_DIR" \
  --alg-config args/alg_args/maac_masked.yaml \
  --save-path "$SAVE_PATH" \
  > "$SAVE_PATH/maac_mask_prior_main_141_${ALIAS}_${VOLTAGE_BARRIER_TYPE}.out" 2>&1 &

# # 4) Ablation: mask + prior bias (explicit s=0.5 config)
# export CUDA_VISIBLE_DEVICES=0
# nohup python -u train_masked_maac.py \
#   --alias "3" \
#   --mode distributed \
#   --scenario case33_3min_final \
#   --voltage-barrier-type "$VOLTAGE_BARRIER_TYPE" \
#   --graph-dir "$GRAPH_DIR" \
#   --alg-config args/alg_args/ablations/maac_masked_mask_prior_add_s050.yaml \
#   --save-path "$SAVE_PATH" \
#   > "$SAVE_PATH/maac_mask_prior_s050_33_${ALIAS}_${VOLTAGE_BARRIER_TYPE}.out" 2>&1 &

# Optional prior-strength sweep
# export CUDA_VISIBLE_DEVICES=0
# nohup python -u train_masked_maac.py \
#   --alias "4" \
#   --mode distributed \
#   --scenario case33_3min_final \
#   --voltage-barrier-type "$VOLTAGE_BARRIER_TYPE" \
#   --graph-dir "$GRAPH_DIR" \
#   --alg-config args/alg_args/ablations/maac_masked_mask_prior_add_s025.yaml \
#   --save-path "$SAVE_PATH" \
#   > "$SAVE_PATH/maac_mask_prior_s025_33_${ALIAS}_${VOLTAGE_BARRIER_TYPE}.out" 2>&1 &

# export CUDA_VISIBLE_DEVICES=0
# nohup python -u train_masked_maac.py \
#   --alias "5" \
#   --mode distributed \
#   --scenario case33_3min_final \
#   --voltage-barrier-type "$VOLTAGE_BARRIER_TYPE" \
#   --graph-dir "$GRAPH_DIR" \
#   --alg-config args/alg_args/ablations/maac_masked_mask_prior_add_s100.yaml \
#   --save-path "$SAVE_PATH" \
#   > "$SAVE_PATH/maac_mask_prior_s100_33_${ALIAS}_${VOLTAGE_BARRIER_TYPE}.out" 2>&1 &

echo "Submitted case33 masked-MAAC experiments."
echo "alias=$ALIAS"
echo "voltage_barrier_type=$VOLTAGE_BARRIER_TYPE"
echo "save_path=$SAVE_PATH"
echo "graph_dir=$GRAPH_DIR"
