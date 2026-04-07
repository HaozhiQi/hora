#!/bin/bash
GPUS=$1
SEED=$2
CACHE=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

echo extra "${EXTRA_ARGS}"

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=AllegroHandHora headless=True seed=${SEED} \
task.env.forceScale=2 task.env.randomForceProbScalar=0.25 \
train.algo=PPO \
task.env.object.type=realshape \
task.env.grasp_cache_name=internal_allegro_realshape \
task.env.numPosePerCache=200 \
task.env.randomization.randomizeScaleList=[0.5,0.52,0.54,0.56,0.58,0.6,0.62,0.64,0.66,0.68,0.7] \
train.ppo.priv_info=True train.ppo.proprio_adapt=False \
train.ppo.output_name=AllegroHandHora/"${CACHE}" \
${EXTRA_ARGS}
