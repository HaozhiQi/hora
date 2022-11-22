# Further Development Guide

In this documentation, we provide explanations to the code structure and starting point for your customized usage.

## File Structure

```
cache/                          # store grasping pose data

configs/
  task/                         # Task and Env Configs
    AllegroHandHora.yaml        # config file for the internal allegro hand, in-hand object rotation
    AllegroHandGrasp.yaml       # config file for the internal allegro hand, generating grasping pose
    PublicAllegroHandHora.yaml  # config file for the public allegro hand, in-hand object rotation
    PublicAllegroHandGrasp.yaml # config file for the public allegro hand, generating grasping pose
  train/                        # PPO Configs
    AllegroHandHora.yaml        # Training PPO for the internal allegro hand
    AllegroHandGrasp.yaml       # for hydra interface use, values are not useful
    PublicAllegroHandHora.yaml  # Training PPO for the public allegro hand
    PublicAllegroHandGrasp.yaml # for hydra interface use, values are not useful
    
hora/                           # main library for in-hand object rotation
  algo/                         # optimization and deployment algorithms
    deploy/
      robots/
        allegro.py              # python interface to the AllegroHand
      deploy.py                 # used for hardware deployment
    models/
      models.py                 # actor critic model definition
      running_mean_std.py       # online normalization class for observations and value estimations
    padapt/
      padapt.py                 # used for stage 2, training adapataion using proprioception
    ppo/
      experience.py             # PPO dataset
      ppo.py                    # PPO trainer
  tasks/
    base/
      vec_task.py               # base task for isaacgym
    allegro_hand_grasp.py       # used for generating grasping poses
    allegro_hand_hora.py        # the main task file for in-hand object rotation
  utils/
    misc.py
    reformat.py

scripts/                        # entry point for running this repository
  deploy.sh                     # hardware deployment script
  eval_s1.sh                    # evaluate stage 1 policy
  eval_s2.sh                    # evaluate stage 2 policy
  train_s1.sh                   # train stage 1 policy
  train_s2.sh                   # train stage 2 policy
  vis_s1.sh                     # visualize stage 1 policy
  vis_s2.sh                     # visualize stage 2 policy
  gen_grasp.sh                  # grasping generation script
```

## Hydra and YAML Arguments

We use [Hydra](https://hydra.cc/docs/intro/) to manage the config.

If you want to change the argument using the command line without modifying the config files, you can simply:

```
scripts/train_s1.sh 0 0 output_name task=PublicAllegroHandHora
```

This will change the task from `AllegroHandHora` to `PublicAllegroHandHora` (using the public allegro).

Default values for each of these are found in the `isaacgymenvs/config/config.yaml` file.

The way that the `task` and `train` portions of the config works are through the use of config groups. You can learn
more about how these work [here](https://hydra.cc/docs/tutorials/structured_config/config_groups/)
The actual configs for `task` are in `isaacgymenvs/config/task/<TASK>.yaml` and for train
in `isaacgymenvs/config/train/<TASK>.yaml`.

In some places in the config you will find other variables referenced (for example,
`num_actors: ${....task.env.numEnvs}`). Each `.` represents going one level up in the config hierarchy. This is
documented fully [here](https://omegaconf.readthedocs.io/en/latest/usage.html#variable-interpolation).

## Important Functions

Compute observation of our task: `compute_observations` in `hora/tasks/allegro_hand_hora.py`.

Compute reward of our task: `compute_reward` in `hora/tasks/allegro_hand_hora.py`.

## Generate the Grasping Pose

Use `scripts/gen_grasp.sh 0 0.8` for generating grasp poses.