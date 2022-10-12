# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on: IsaacGymEnvs
# Copyright (c) 2018-2022, NVIDIA Corporation
# Licence under BSD 3-Clause License
# https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/
# --------------------------------------------------------

from isaacgym import gymapi

import abc
import sys
import gym
import torch
import numpy as np

from abc import ABC
from gym import spaces
from typing import Dict, Any, Tuple


class Env(ABC):
    def __init__(self, config: Dict[str, Any], sim_device: str, graphics_device_id: int,  headless: bool):
        """Initialise the env.

        Args:
            config: the configuration dictionary.
            sim_device: the device to simulate physics on. eg. 'cuda:0' or 'cpu'
            graphics_device_id: the device ID to render with.
            headless: Set to False to disable viewer rendering.
        """

        split_device = sim_device.split(':')
        self.device_type = split_device[0]
        self.device_id = int(split_device[1]) if len(split_device) > 1 else 0

        self.device = 'cpu'
        if config['sim']['use_gpu_pipeline']:
            if self.device_type.lower() == 'cuda' or self.device_type.lower() == 'gpu':
                self.device = 'cuda' + ':' + str(self.device_id)
            else:
                print('GPU Pipeline can only be used with GPU simulation. Forcing CPU Pipeline.')
                config['sim']['use_gpu_pipeline'] = False

        self.rl_device = config.get('rl_device', 'cuda:0')

        # Rendering
        # if training in a headless mode
        self.headless = headless

        enable_camera_sensors = config.get('enableCameraSensors', False)
        self.graphics_device_id = graphics_device_id
        if not enable_camera_sensors and self.headless:
            self.graphics_device_id = -1

        self.num_environments = config['env']['numEnvs']
        self.num_observations = config['env']['numObservations']
        self.num_actions = config['env']['numActions']

        self.obs_space = spaces.Box(np.ones(self.num_obs, dtype=np.float32) * -np.Inf, np.ones(self.num_obs, dtype=np.float32) * np.Inf)
        self.act_space = spaces.Box(np.ones(self.num_actions, dtype=np.float32) * -1., np.ones(self.num_actions, dtype=np.float32) * 1.)

        self.clip_obs = config['env'].get('clipObservations', np.Inf)
        self.clip_actions = config['env'].get('clipActions', np.Inf)

        # controller
        controller_config = config['env']['controller']
        self.torque_control = controller_config['torque_control']
        self.p_gain = controller_config['pgain']
        self.d_gain = controller_config['dgain']
        self.control_freq_inv = controller_config['controlFrequencyInv']

    @abc.abstractmethod 
    def _allocate_buffers(self):
        """Create torch buffers for observations, rewards, actions dones and any additional data."""

    @abc.abstractmethod
    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """

    @abc.abstractmethod
    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset the environment.
        Returns:
            Observation dictionary
        """

    @property
    def observation_space(self) -> gym.Space:
        """Get the environment's observation space."""
        return self.obs_space

    @property
    def action_space(self) -> gym.Space:
        """Get the environment's action space."""
        return self.act_space

    @property
    def num_envs(self) -> int:
        """Get the number of environments."""
        return self.num_environments

    @property
    def num_acts(self) -> int:
        """Get the number of actions in the environment."""
        return self.num_actions

    @property
    def num_obs(self) -> int:
        """Get the number of observations in the environment."""
        return self.num_observations


class VecTask(Env):

    def __init__(self, config, sim_device, graphics_device_id, headless):
        """Initialise the `VecTask`.

        Args:
            config: config dictionary for the environment.
            sim_device: the device to simulate physics on. eg. 'cuda:0' or 'cpu'
            graphics_device_id: the device ID to render with.
            headless: Set to False to disable viewer rendering.
        """
        super().__init__(config, sim_device, graphics_device_id, headless)

        self.sim_params = self._parse_sim_params(config['physics_engine'], config['sim'])
        if config['physics_engine'] == 'physx':
            self.physics_engine = gymapi.SIM_PHYSX
        elif config['physics_engine'] == 'flex':
            self.physics_engine = gymapi.SIM_FLEX
        else:
            msg = f"Invalid physics engine backend: {config['physics_engine']}"
            raise ValueError(msg)

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)  # noqa
        torch._C._jit_set_profiling_executor(False)  # noqa

        self.gym = gymapi.acquire_gym()
        self._allocate_buffers()
        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)
        self._set_viewer()
        self.obs_dict = {}

    def _set_viewer(self):
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if not self.headless:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, 'QUIT')
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, 'toggle_viewer_sync')

            # set the camera position based on up axis
            sim_params = self.gym.get_sim_params(self.sim)
            if sim_params.up_axis == gymapi.UP_AXIS_Z:
                cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
                cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
            else:
                cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
                cam_target = gymapi.Vec3(10.0, 0.0, 15.0)

            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _allocate_buffers(self):
        """Allocate the observation, states, etc. buffers.

        These are what is used to set observations and states in the environment classes which
        inherit from this one, and are read in `step` and other related functions.

        """
        # allocate buffers
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        self.obs_buf_lag_history = torch.zeros((
            self.num_envs, 80, self.num_obs // 3
        ), device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.at_reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.timeout_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}
        self._allocate_task_buffer(self.num_envs)

    def _allocate_task_buffer(self, num_envs):
        pass

    def set_sim_params_up_axis(self, sim_params: gymapi.SimParams, axis: str) -> int:
        """Set gravity based on up axis and return axis index.

        Args:
            sim_params: sim params to modify the axis for.
            axis: axis to set sim params for.
        Returns:
            axis index for up axis.
        """
        if axis == 'z':
            sim_params.up_axis = gymapi.UP_AXIS_Z
            sim_params.gravity.x = 0
            sim_params.gravity.y = 0
            sim_params.gravity.z = -9.81
            return 2
        return 1

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)
        self.sim = self.gym.create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        if self.sim is None:
            print('*** Failed to create sim')
            quit()
        self._create_envs(self.num_envs, self.config['env']['envSpacing'], int(np.sqrt(self.num_envs)))

    @abc.abstractmethod
    def _create_envs(self, num_envs, spacing, num_per_row):
        """Create Training Environments
        Args:
            num_envs: number of parallel environments
            spacing: space between different envs
            num_per_row:
        """

    @abc.abstractmethod
    def pre_physics_step(self, actions: torch.Tensor):
        """Apply the actions to the environment (eg by setting torques, position targets).

        Args:
            actions: the actions to apply
        """

    @abc.abstractmethod
    def post_physics_step(self):
        """Compute reward and observations, reset any environments that require it."""

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """
        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # apply actions
        self.pre_physics_step(action_tensor)

        # step physics and render each frame
        for i in range(self.control_freq_inv):
            self.render()
            self.update_low_level_control()
            self.gym.simulate(self.sim)

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)

        # fill time out buffer
        self.timeout_buf = torch.where(
            torch.greater_equal(self.progress_buf, self.max_episode_length - 1),
            torch.ones_like(self.timeout_buf), torch.zeros_like(self.timeout_buf)
        )

        # compute observations, rewards, resets, ...
        self.post_physics_step()
        self.extras['time_outs'] = self.timeout_buf.to(self.rl_device)
        self.obs_dict['obs'] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras

    def update_low_level_control(self):
        pass

    def zero_actions(self) -> torch.Tensor:
        """Returns a buffer with zero actions.

        Returns:
            A buffer of zero torch actions
        """
        actions = torch.zeros([self.num_envs, self.num_actions], dtype=torch.float32, device=self.rl_device)

        return actions

    def reset(self) -> Dict[str, torch.Tensor]:
        """Reset the environment.
        Returns:
            Observation dictionary
        """
        env_ids = self.reset_buf.nonzero().squeeze(-1)
        self.reset_idx(env_ids)
        zero_actions = self.zero_actions()
        # step the simulator
        self.step(zero_actions)
        self.obs_dict['obs'] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        return self.obs_dict

    def reset_idx(self, env_ids):
        raise NotImplementedError

    def render(self):
        """Draw the frame to the viewer, and check for keyboard events."""
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == 'QUIT' and evt.value > 0:
                    sys.exit()
                elif evt.action == 'toggle_viewer_sync' and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
                # Wait for dt to elapse in real time.
                # This synchronizes the physics simulation with the rendering rate.
                self.gym.sync_frame_time(self.sim)
            else:
                self.gym.poll_viewer_events(self.viewer)

    def _parse_sim_params(self, physics_engine: str, config_sim: Dict[str, Any]) -> gymapi.SimParams:
        """Parse the config dictionary for physics stepping settings.

        Args:
            physics_engine: which physics engine to use. "physx" or "flex"
            config_sim: dict of sim configuration parameters
        Returns
            IsaacGym SimParams object with updated settings.
        """
        sim_params = gymapi.SimParams()

        # check correct up-axis
        if config_sim['up_axis'] not in ['z', 'y']:
            msg = f"Invalid physics up-axis: {config_sim['up_axis']}"
            print(msg)
            raise ValueError(msg)

        # assign general sim parameters
        sim_params.dt = config_sim['dt']
        sim_params.num_client_threads = config_sim.get('num_client_threads', 0)
        sim_params.use_gpu_pipeline = config_sim['use_gpu_pipeline']
        sim_params.substeps = config_sim.get('substeps', 2)

        # assign up-axis
        if config_sim['up_axis'] == 'z':
            sim_params.up_axis = gymapi.UP_AXIS_Z
        else:
            sim_params.up_axis = gymapi.UP_AXIS_Y

        # assign gravity
        sim_params.gravity = gymapi.Vec3(*config_sim['gravity'])

        # configure physics parameters
        if physics_engine == 'physx':
            # set the parameters
            if 'physx' in config_sim:
                for opt in config_sim['physx'].keys():
                    if opt == 'contact_collection':
                        setattr(sim_params.physx, opt, gymapi.ContactCollection(config_sim['physx'][opt]))
                    else:
                        setattr(sim_params.physx, opt, config_sim['physx'][opt])
        else:
            # set the parameters
            if 'flex' in config_sim:
                for opt in config_sim['flex'].keys():
                    setattr(sim_params.flex, opt, config_sim['flex'][opt])

        return sim_params
