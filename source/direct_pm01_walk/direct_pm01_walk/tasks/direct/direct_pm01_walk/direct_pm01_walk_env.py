# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .direct_pm01_walk_env_cfg import DirectPm01WalkEnvCfg
from direct_pm01_walk.tasks.direct.direct_pm01_walk.rewards.rewards import *
from isaaclab.utils.math import quat_apply



class DirectPm01WalkEnv(DirectRLEnv):
    cfg: DirectPm01WalkEnvCfg

    def __init__(self, cfg: DirectPm01WalkEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.default_joint_pos = self.robot.data.default_joint_pos.clone()


    def _setup_scene(self):
        self.robot = Articulation(self.cfg.scene.robot)
        self.scene.articulations["robot"] = self.robot


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        action_scale = 1.0
        joint_target = self.default_joint_pos + self.actions * action_scale
        self.robot.set_joint_position_target(joint_target)


    def _get_observations(self) -> dict:
        base_lin_vel = self.robot.data.root_lin_vel_b
        base_ang_vel = self.robot.data.root_ang_vel_b
        joint_pos = self.robot.data.joint_pos
        joint_vel = self.robot.data.joint_vel

        base_quat = self.robot.data.root_quat_w
        num_envs = base_quat.shape[0]
        gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float32).repeat(num_envs, 1)
        projected_gravity = quat_apply(base_quat, gravity_vec)  # (num_envs, 3)

        obs = torch.cat([base_lin_vel, base_ang_vel, joint_pos, joint_vel, projected_gravity], dim=-1)

        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        l2 = flat_orientation_l2(self)  # 传入 env
        #print("flat_orientation_l2:", l2.mean().item())
        reward = -l2

        penalty = fall_penalty(self)
        #print("fall_penalty:", penalty.mean().item())
        reward -= penalty

        joint_pos_limits_penalty = joint_pos_limits(self)
        #print("joint_pos_limits_penalty:", joint_pos_limits_penalty.mean().item())
        reward -= joint_pos_limits_penalty * 0.1

        joint_torques_penalty = joint_torques_l2(self)
        #print("joint_torques_penalty:", joint_torques_penalty.mean().item())
        reward -= joint_torques_penalty * 0.01

        joint_acc_penalty = joint_acc_l2(self)
        #print("joint_acc_penalty:", joint_acc_penalty.mean().item())
        reward -= joint_acc_penalty * 0.00000001

        action_rate_penalty = action_rate_l2(self)
        #print("action_rate_penalty:", action_rate_penalty.mean().item())
        reward -= action_rate_penalty * 0.01

        lin_vel_z_penalty = lin_vel_z_l2(self)
        #print("lin_vel_z_penalty:", lin_vel_z_penalty.mean().item())
        reward -= lin_vel_z_penalty * 0.01

        ang_vel_xy_penalty = ang_vel_xy_l2(self)
        #print("ang_vel_xy_penalty:", ang_vel_xy_penalty.mean().item())
        reward -= ang_vel_xy_penalty * 0.01

        #print("total reward:", reward.mean().item())
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        fallen = self.robot.data.root_pos_w[:, 2] < 0.4
        done = fallen
        return done, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset selected environments to default state (minimal version)."""
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        # 调用父类逻辑（清理 buffers）
        super()._reset_idx(env_ids)

        # 默认状态
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        #print("joint pos on reset:", joint_pos[0])
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        root_state = self.robot.data.default_root_state[env_ids].clone()
        #print("root state on reset:", root_state[0])

        # 将每个环境放到对应的 origin（env_spacing 控制）
        root_state[:, :3] += self.scene.env_origins[env_ids]

        #print("root state after setting origin:", root_state[0])

        # # 姿态添加少量随机扰动
        # quat = torch.tensor([1, 0, 0, 0], device=self.device).repeat(len(env_ids), 1)
        # noise_axis = torch.randn_like(quat[:, 1:])
        # noise_axis = noise_axis / torch.norm(noise_axis, dim=-1, keepdim=True)
        # noise_angle = 0.05 * torch.randn(len(env_ids), 1, device=self.device)  # 约3度随机旋转
        # sin_half = torch.sin(noise_angle / 2)
        # quat_noise = torch.cat([torch.cos(noise_angle / 2), sin_half * noise_axis], dim=-1)
        # root_state[:, 3:7] = quat_noise

        # #base质心添加少量随机扰动
        # com_offset = 0.01 * torch.randn(len(env_ids), 3, device=self.device)
        # self.robot.data.com_pos_w[env_ids] += com_offset

        # 写入仿真
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

