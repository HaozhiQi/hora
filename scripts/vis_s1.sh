#!/bin/bash
CACHE=$1
python train.py task=AllegroHandHora headless=False pipeline=gpu \
task.env.numEnvs=1 test=True \
task.env.object.type=simple_tennis_ball \
train.algo=PPO \
task.env.randomization.randomizeMass=False \
task.env.randomization.randomizeCOM=False \
task.env.randomization.randomizeFriction=False \
task.env.randomization.randomizePDGains=False \
task.env.randomization.randomizeScale=True \
train.ppo.priv_info=True \
train.ppo.output_name=AllegroHandHora/"${CACHE}" \
checkpoint=outputs/AllegroHandHora/"${CACHE}"/stage1_nn/best.pth