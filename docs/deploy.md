# Deployment Instructions

This document provides instructions on how to deploy a trained policy on the real AllegroHand.

## Install ROS

This section provides instruction for installing ROS. Note that there are varies ways of installing ROS. Since this project is developed using Ubuntu 22.04, which is not officially supported by ROS. During the development of this project, we install ros from [robostack](https://github.com/RoboStack/ros-noetic). It's ok to ignore this section if you already have a working ROS library setup.

The following instructions are copied from [robostack](https://github.com/RoboStack/ros-noetic). We assume you have a miniconda or conda environment installed.

First, install `mamba` in the `base` environment.

```
# if you don't have mamba yet, install it first in the base environment (not needed when using mambaforge):
conda install mamba -c conda-forge
```

Then, install `rosallegro` conda environment. Note that his `rosallegro` environment is different from what we used to train the policy. This is mainly to solve possible conflicts between ros and varies packages we used (e.g. `pytorch`).

```
mamba create -n rosallegro ros-noetic-desktop python=3.9 -c robostack -c robostack-experimental -c conda-forge --no-channel-priority --override-channels
conda activate rosallegro
```

Finally, follow the instructions at [ros-allegro](https://github.com/HaozhiQi/ros-allegro/) repository to set up the ROS package and the PCAN driver.

## Usage

First, you need to open a shell and launch the following script:
```
# may need to change /dev/pcanusbfd32 to your customized values
roslaunch allegro_hand allegro_hand.launch HAND:=right AUTO_CAN:=false CAN_DEVICE:=/dev/pcanusbfd32 KEYBOARD:=false CONTROLLER:=pd
```

Once it's launched, run the following command in the `hora` conda environment to control the robot hand.
```
# e.g. scripts/deploy.sh hora
conda activate hora
scripts/deploy.sh ${OUTPUT_NAME}
```
