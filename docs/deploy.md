# Deployment Instructions

In this document, we provide instructions of how to deploy the model on a real AllegroHand.

(This document is still under active polishing and construction, ping me in discord if you need help or want to contribute).

## Install ROS

There are varies ways of installing ROS. In my experiments, I found the official ROS release is not compatible with my operating system. I need to use the following workaround.
```
conda create -y -n deploy python=3.8
conda install -c conda-forge -c robostack ros-noetic-desktop
```

## Install PCAN Driver

```
# in peak-linux-driver
make clean
make NET=NO_NETDEV_SUPPORT
sudo make install
sudo /sbin/modprobe pcan
```

## Install This Repository

The reason I use -j1 is because I don't know how to resolve the dependency error when doing parallel build. It seems to be the libpcan problem. However, I could not simply install libpcan maybe because the ros is installed from robostack.

```
cd ${YOUR_CATKIN_WORKSPACE}/src/
git clone https://HaozhiQi/allegro-deploy.git
cd ${YOUR_CATKIN_WORKSPACE}/
catkin_make -j1
catkin_make install
source ./devel/setup.zsh # or other shell name
```

## Usage

```
# may need to change /dev/pcanusbfd32 to your customized values
roslaunch allegro_hand allegro_hand.launch HAND:=right AUTO_CAN:=false CAN_DEVICE:=/dev/pcanusbfd32 KEYBOARD:=false
```