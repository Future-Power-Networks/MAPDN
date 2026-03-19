#!/usr/bin/env bash
if [ ! -d "$3" ]; then
  mkdir -p "$3"
fi

export CUDA_VISIBLE_DEVICES=0
nohup bash -c '
echo "start_time: $(date)"
time python -u train.py \
  --alg maac \
  --alias '"$1"' \
  --mode distributed \
  --scenario case33_3min_final \
  --voltage-barrier-type '"$2"' \
  --save-path '"$3"'
echo "end_time: $(date)"
' > out/maac_33_$1_$2.out 2>&1 &

export CUDA_VISIBLE_DEVICES=0
nohup bash -c '
echo "start_time: $(date)"
time python -u train.py \
  --alg maac \
  --alias '"$1"' \
  --mode distributed \
  --scenario case141_3min_final \
  --voltage-barrier-type '"$2"' \
  --save-path '"$3"'
echo "end_time: $(date)"
' > out/maac_141_$1_$2.out 2>&1 &
