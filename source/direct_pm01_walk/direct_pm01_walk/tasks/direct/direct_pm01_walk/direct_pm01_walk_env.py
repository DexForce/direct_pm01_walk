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
        self.gait_phase = torch.zeros(self.num_envs, device=self.device)

        self._lfoot_ids, _ = self.robot.find_bodies("link_ankle_roll_l")
        self._rfoot_ids, _ = self.robot.find_bodies("link_ankle_roll_r")
        assert len(self._lfoot_ids) == 1 and len(self._rfoot_ids) == 1, "检查脚链路命名是否匹配"
        self._l = self._lfoot_ids[0]
        self._r = self._rfoot_ids[0]


    def _setup_scene(self):
        self.robot = Articulation(self.cfg.scene.robot)
        self.scene.articulations["robot"] = self.robot


    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        phase_delta = 2 * math.pi * self.cfg.sim.dt / 0.8  #周期为0.8秒
        self.gait_phase = (self.gait_phase + phase_delta) % (2 * math.pi)

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


    # 左腿：j00_hip_pitch_l、j01_hip_roll_l、j02_hip_yaw_l、j03_knee_pitch_l、j04_ankle_pitch_l、j05_ankle_roll_l。
    # 右腿：j06_hip_pitch_r、j07_hip_roll_r、j08_hip_yaw_r、j09_knee_pitch_r、j10_ankle_pitch_r、j11_ankle_roll_r。
    # 躯干：j12_waist_yaw。
    # 左臂：j13_shoulder_pitch_l、j14_shoulder_roll_l、j15_shoulder_yaw_l、j16_elbow_pitch_l、j17_elbow_yaw_l。
    # 右臂：j18_shoulder_pitch_r、j19_shoulder_roll_r、j20_shoulder_yaw_r、j21_elbow_pitch_r、j22_elbow_yaw_r。
    # 头部：j23_head_yaw。

    def _get_rewards(self) -> torch.Tensor:
        l2 = flat_orientation_l2(self)  # 传入 env
        weight = 1.0
        print("flat_orientation_l2: %.3f \t weighted: %.3f" % (-l2.mean().item(), -l2.mean().item() * weight))
        reward = -l2*weight

        penalty = fall_penalty(self)
        weight = 1.0
        print("fall_penalty: %.3f \t weighted: %.3f" % (-penalty.mean().item(), -penalty.mean().item() * weight))
        reward -= penalty * weight

        joint_pos_limits_penalty = joint_pos_limits(self)
        weight = 0.1
        print("joint_pos_limits_penalty: %.3f \t weighted: %.3f" % (-joint_pos_limits_penalty.mean().item(), -joint_pos_limits_penalty.mean().item() * weight))
        reward -= joint_pos_limits_penalty * weight

        joint_torques_penalty = joint_torques_l2(self)
        weight = 0.01
        print("joint_torques_penalty: %.3f \t weighted: %.3f" % (-joint_torques_penalty.mean().item(), -joint_torques_penalty.mean().item() * weight))
        reward -= joint_torques_penalty * weight

        joint_acc_penalty = joint_acc_l2(self)
        weight = 0.00000001
        print("joint_acc_penalty: %.3f \t weighted: %.3f" % (-joint_acc_penalty.mean().item(), -joint_acc_penalty.mean().item() * weight))
        reward -= joint_acc_penalty * weight

        action_rate_penalty = action_rate_l2(self)
        weight = 0.01
        print("action_rate_penalty: %.3f \t weighted: %.3f" % (-action_rate_penalty.mean().item(), -action_rate_penalty.mean().item() * weight))
        reward -= action_rate_penalty * weight

        lin_vel_z_penalty = lin_vel_z_l2(self)
        weight = 0.01
        print("lin_vel_z_penalty: %.3f \t weighted: %.3f" % (-lin_vel_z_penalty.mean().item(), -lin_vel_z_penalty.mean().item() * weight))
        reward -= lin_vel_z_penalty * weight

        ang_vel_xy_penalty = ang_vel_xy_l2(self)
        weight = 0.01
        print("ang_vel_xy_penalty: %.3f \t weighted: %.3f" % (-ang_vel_xy_penalty.mean().item(), -ang_vel_xy_penalty.mean().item() * weight))
        reward -= ang_vel_xy_penalty * weight

        gait_phase_reward = get_gait_phase_reward(self)
        weight = 5.0
        print("gait_phase_reward: %.3f \t weighted: %.3f" % (gait_phase_reward.mean().item(), gait_phase_reward.mean().item() * weight))
        reward += gait_phase_reward * weight
        
        upper_body_deviation_penalty = joint_deviation_l1(self, 
                                                         joint_names=["j12_waist_yaw",
                                                                      "j13_shoulder_pitch_l", "j14_shoulder_roll_l", "j15_shoulder_yaw_l",
                                                                      "j16_elbow_pitch_l", "j17_elbow_yaw_l",
                                                                      "j18_shoulder_pitch_r", "j19_shoulder_roll_r", "j20_shoulder_yaw_r",
                                                                      "j21_elbow_pitch_r", "j22_elbow_yaw_r",
                                                                      "j23_head_yaw"])
        weight = 0.1
        reward -= upper_body_deviation_penalty * weight
        print("upper_body_deviation_penalty: %.3f \t weighted: %.3f" % (-upper_body_deviation_penalty.mean().item(), -upper_body_deviation_penalty.mean().item() * weight))

        hip_deviation_penalty = joint_deviation_l1(self, joint_names=["j02_hip_yaw_l", "j08_hip_yaw_r", "j01_hip_roll_l", "j07_hip_roll_r"])
        weight = 1.0
        reward -= hip_deviation_penalty * weight
        print("hip_deviation_penalty: %.3f \t weighted: %.3f" % (-hip_deviation_penalty.mean().item(), -hip_deviation_penalty.mean().item() * weight))

        leg_deviation_penalty = joint_deviation_l1(self, 
                                                   joint_names=["j00_hip_pitch_l", "j06_hip_pitch_r",
                                                                "j03_knee_pitch_l", "j09_knee_pitch_r",
                                                                "j04_ankle_pitch_l", "j10_ankle_pitch_r"])
        weight = 0.001
        reward -= leg_deviation_penalty * weight
        print("leg_deviation_penalty: %.3f \t weighted: %.3f" % (-leg_deviation_penalty.mean().item(), -leg_deviation_penalty.mean().item() * weight))

        joint_symmetry_penalty = joint_symmetry_l2(self, 
                                                   joint_pairs=[
                                                       ["j00_hip_pitch_l", "j06_hip_pitch_r"],
                                                       ["j03_knee_pitch_l", "j09_knee_pitch_r"],
                                                       ["j04_ankle_pitch_l", "j10_ankle_pitch_r"],
                                                   ])
        weight = 0.1
        reward -= joint_symmetry_penalty * weight
        print("joint_symmetry_penalty: %.3f \t weighted: %.3f" % (-joint_symmetry_penalty.mean().item(), -joint_symmetry_penalty.mean().item() * weight))

        print("total reward: %.3f" % reward.mean().item())
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

        # 姿态添加少量随机扰动
        quat = torch.ones((len(env_ids), 4), device=self.device, dtype=torch.float32)
        quat[:, 1:] = 0.0
        noise_axis = torch.randn_like(quat[:, 1:])
        noise_axis = noise_axis / torch.norm(noise_axis, dim=-1, keepdim=True)
        noise_angle = 0.15 * torch.randn(len(env_ids), 1, device=self.device)  # 约9度随机旋转
        sin_half = torch.sin(noise_angle / 2)
        quat_noise = torch.cat([torch.cos(noise_angle / 2), sin_half * noise_axis], dim=-1)
        root_state[:, 3:7] = quat_noise


        # 写入仿真
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.gait_phase[env_ids] = 0.0