#!/bin/bash
GPUS=$1
CACHE=$2

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:2:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo extra "${EXTRA_ARGS}"

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=AllegroHandHora headless=False pipeline=gpu \
task.env.numEnvs=1 test=True \
task.env.object.type=simple_tennis_ball \
task.env.randomization.randomizeMass=False \
task.env.randomization.randomizeCOM=False \
task.env.randomization.randomizeFriction=False \
task.env.randomization.randomizePDGains=False \
task.env.randomization.randomizeScale=True \
train.algo=ProprioAdapt \
train.ppo.priv_info=True train.ppo.proprio_adapt=True \
train.ppo.output_name=AllegroHandHora/"${CACHE}" \
checkpoint=outputs/AllegroHandHora/"${CACHE}"/stage2_nn/model_last.ckpt \
${EXTRA_ARGS}
