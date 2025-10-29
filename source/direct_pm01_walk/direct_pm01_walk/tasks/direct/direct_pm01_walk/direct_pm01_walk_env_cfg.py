# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#from isaaclab_assets.robots.cartpole import CARTPOLE_CFG

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from direct_pm01_walk.assets.robots.pm01.pm01 import PM01_CFG
import isaaclab.sim as sim_utils
from isaaclab.sensors import ContactSensorCfg


import gymnasium as gym
import numpy as np


@configclass
class Pm01WalkSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # robot
    robot: ArticulationCfg = PM01_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot", 
        spawn=PM01_CFG.spawn.replace(activate_contact_sensors=True),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", 
        history_length=3, 
        track_air_time=True
    )


@configclass
class DirectPm01WalkEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 30.0

    # - spaces definition
    action_space = 24
    observation_space = 57
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation)

    # scene
    scene: Pm01WalkSceneCfg = Pm01WalkSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)
