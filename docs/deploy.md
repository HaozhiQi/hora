# Deployment Instructions

In this document, we provide instructions of how to deploy the model on a real AllegroHand.

(This document is still under active polishing and construction, ping me in discord if you need help or want to contribute).

## Install ROS

There are varies ways of installing ROS. During the development of this project, we use the following command to install ROS:
```
conda install -c conda-forge -c robostack ros-noetic-desktop
```
This is because of the official ROS release does not support my operating system at that time.

## Setup the Allegro ROS library

Follow the insturctions at [ros-allegro](https://github.com/HaozhiQi/ros-allegro/) repository to setup the ROS package and the PCAN driver.

## Usage

First, you need to open a shell and launch the following script:
```
# may need to change /dev/pcanusbfd32 to your customized values
roslaunch allegro_hand allegro_hand.launch HAND:=right AUTO_CAN:=false CAN_DEVICE:=/dev/pcanusbfd32 KEYBOARD:=false
```
Once it's launched, run the following command to control the robot hand.
```
# e.g. scripts/deploy.sh hora
scripts/deploy.sh ${OUTPUT_NAME}
```
