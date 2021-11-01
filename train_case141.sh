if [ ! -d $3 ]
then
  mkdir $3
fi


export CUDA_VISIBLE_DEVICES=1
nohup python train.py --alg coma --alias $1 --mode distributed --scenario case141_3min_final --voltage-barrier-type $2 --save-path $3 > coma_141_$1_$2.out &
export CUDA_VISIBLE_DEVICES=1
nohup python train.py --alg iddpg --alias $1 --mode distributed --scenario case141_3min_final --voltage-barrier-type $2 --save-path $3 > iddpg_141_$1_$2.out &
export CUDA_VISIBLE_DEVICES=1
nohup python train.py --alg ippo --alias $1 --mode distributed --scenario case141_3min_final --voltage-barrier-type $2 --save-path $3 > ippo_141_$1_$2.out &
export CUDA_VISIBLE_DEVICES=1
nohup python train.py --alg matd3 --alias $1 --mode distributed --scenario case141_3min_final --voltage-barrier-type $2 --save-path $3 > matd3_141_$1_$2.out &
export CUDA_VISIBLE_DEVICES=0
nohup python train.py --alg maddpg --alias $1 --mode distributed --scenario case141_3min_final --voltage-barrier-type $2 --save-path $3 > maddpg_141_$1_$2.out &
export CUDA_VISIBLE_DEVICES=0
nohup python train.py --alg mappo --alias $1 --mode distributed --scenario case141_3min_final --voltage-barrier-type $2 --save-path $3 > mappo_141_$1_$2.out &
# export CUDA_VISIBLE_DEVICES=1
# nohup python train.py --alg maac --alias $1 --mode distributed --scenario case141_3min_final --voltage-barrier-type $2 --save-path $3 > maac_141_$1_$2.out &
export CUDA_VISIBLE_DEVICES=0
nohup python train.py --alg sqddpg --alias $1 --mode distributed --scenario case141_3min_final --voltage-barrier-type $2 --save-path $3 > sqddpg_141_$1_$2.out &