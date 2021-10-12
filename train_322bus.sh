export CUDA_VISIBLE_DEVICES=0
nohup python train.py --alg coma --alias $1 --difficulty $2 --mode distributed --scenario bus322_gu_3min --reward-type $3 --save-path one_day > coma_322_$1_$2_$3.out &
export CUDA_VISIBLE_DEVICES=0
nohup python train.py --alg iddpg --alias $1 --difficulty $2 --mode distributed --scenario bus322_gu_3min --reward-type $3 --save-path one_day > iddpg_322_$1_$2_$3.out &
export CUDA_VISIBLE_DEVICES=0
nohup python train.py --alg ippo --alias $1 --difficulty $2 --mode distributed --scenario bus322_gu_3min --reward-type $3 --save-path one_day > ippo_322_$1_$2_$3.out &
# export CUDA_VISIBLE_DEVICES=0
# nohup python train.py --alg maac --alias $1 --difficulty $2 --mode distributed --scenario bus322_gu_3min --reward-type $3 > maac_322_$1_$2_$3.out &
export CUDA_VISIBLE_DEVICES=1
nohup python train.py --alg maddpg --alias $1 --difficulty $2 --mode distributed --scenario bus322_gu_3min --reward-type $3 --save-path one_day > maddpg_322_$1_$2_$3.out &
export CUDA_VISIBLE_DEVICES=1
nohup python train.py --alg mappo --alias $1 --difficulty $2 --mode distributed --scenario bus322_gu_3min --reward-type $3 --save-path one_day > mappo_322_$1_$2_$3.out &
export CUDA_VISIBLE_DEVICES=0
nohup python train.py --alg matd3 --alias $1 --difficulty $2 --mode distributed --scenario bus322_gu_3min --reward-type $3 --save-path one_day > matd3_322_$1_$2_$3.out &
export CUDA_VISIBLE_DEVICES=1
nohup python train.py --alg sqddpg --alias $1 --difficulty $2 --mode distributed --scenario bus322_gu_3min --reward-type $3 --save-path one_day > sqddpg_322_$1_$2_$3.out &