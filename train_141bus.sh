export CUDA_VISIBLE_DEVICES=0
nohup python train.py --alg coma --alias $1 --difficulty $2 --mode distributed --scenario bus141_gu_3min --reward-type $3 > coma_141_$1_$2_$3.out &
export CUDA_VISIBLE_DEVICES=0
nohup python train.py --alg iddpg --alias $1 --difficulty $2 --mode distributed --scenario bus141_gu_3min --reward-type $3 > iddpg_141_$1_$2_$3.out &
export CUDA_VISIBLE_DEVICES=0
nohup python train.py --alg ippo --alias $1 --difficulty $2 --mode distributed --scenario bus141_gu_3min --reward-type $3 > ippo_141_$1_$2_$3.out &
# export CUDA_VISIBLE_DEVICES=0
# nohup python train.py --alg matd3 --alias $1 --difficulty $2 --mode distributed --scenario bus141_gu_3min --reward-type $3 > matd3_141_$1_$2_$3.out &
export CUDA_VISIBLE_DEVICES=1
nohup python train.py --alg maddpg --alias $1 --difficulty $2 --mode distributed --scenario bus141_gu_3min --reward-type $3 > maddpg_141_$1_$2_$3.out &
export CUDA_VISIBLE_DEVICES=1
nohup python train.py --alg mappo --alias $1 --difficulty $2 --mode distributed --scenario bus141_gu_3min --reward-type $3 > mappo_141_$1_$2_$3.out &
# export CUDA_VISIBLE_DEVICES=1
# nohup python train.py --alg maac --alias $1 --difficulty $2 --mode distributed --scenario bus141_gu_3min --reward-type $3 > maac_141_$1_$2_$3.out &
export CUDA_VISIBLE_DEVICES=1
nohup python train.py --alg sqddpg --alias $1 --difficulty $2 --mode distributed --scenario bus141_gu_3min --reward-type $3 > sqddpg_141_$1_$2_$3.out &