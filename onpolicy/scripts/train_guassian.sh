#!/bin/sh
env="gaussian"
scenario="cgs"  # simple_speaker_listener # simple_spread
num_landmarks=3
num_agents=3
algo="mappo"
exp="mlp"
seed_max=5

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    CUDA_VISIBLE_DEVICES=1 python train/train_guassian.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${scenario} --seed 1 --n_actions 19 --n_training_threads 127 --n_rollout_threads 8 --num_agents 3 --num_mini_batch 1 --episode_length 210 --num_env_steps 10000000 --ppo_epoch 5 --use_value_active_masks --add_center_xy --use_wandb --use_state_agent 
done
