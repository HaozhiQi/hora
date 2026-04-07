#!/bin/bash
# Generate per-object grasp cache for realshape objects
# 200 grasps per object × 180 objects = 36000 grasps per scale
# Usage: bash scripts/gen_grasp_realshape.sh GPU_ID SCALE
# Example: bash scripts/gen_grasp_realshape.sh 0 0.5
GPUS=$1
SCALE=$2
CUDA_VISIBLE_DEVICES=${GPUS} \
python gen_grasp.py task=AllegroHandGrasp headless=True pipeline=cpu \
task.env.numEnvs=20000 test=True \
task.env.controller.controlFrequencyInv=8 task.env.episodeLength=50 \
task.env.controller.torque_control=False task.env.genGrasps=True task.env.baseObjScale="${SCALE}" \
task.env.object.type=realshape \
task.env.grasp_cache_name=internal_allegro_realshape \
task.env.numPosePerCache=200 \
task.env.randomization.randomizeMass=True task.env.randomization.randomizeMassLower=0.05 task.env.randomization.randomizeMassUpper=0.051 \
task.env.randomization.randomizeCOM=False \
task.env.randomization.randomizeFriction=False \
task.env.randomization.randomizePDGains=False \
task.env.randomization.randomizeScale=False \
train.ppo.priv_info=True
