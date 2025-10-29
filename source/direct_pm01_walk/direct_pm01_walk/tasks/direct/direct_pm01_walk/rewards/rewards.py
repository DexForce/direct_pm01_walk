from isaaclab.utils.math import quat_apply
import torch

def _get_sim_dt(env) -> float:
    """尝试获取仿真步长，获取失败时返回 1.0 作为兜底。"""

    if hasattr(env, "step_dt"):
        return float(env.step_dt)
    if hasattr(env, "_step_dt"):
        return float(env._step_dt)
    if hasattr(env, "cfg") and getattr(env.cfg, "sim", None) is not None:
        sim_cfg = env.cfg.sim
        if hasattr(sim_cfg, "dt"):
            return float(sim_cfg.dt)
    return 1.0


def flat_orientation_l2(env):
    """计算与世界Z轴的偏离程度，鼓励身体保持直立"""
    base_quat = env.robot.data.root_quat_w   # (num_envs, 4)

    num_envs = base_quat.shape[0]
    world_up = torch.tensor([0.0, 0.0, 1.0], device=env.device, dtype=torch.float32).repeat(num_envs, 1)
    up_vector = quat_apply(base_quat, world_up)  # (num_envs, 3)

    # 惩罚身体倾斜：up_vector 应该尽量接近 [0, 0, 1]
    deviation = 1.0 - up_vector[:, 2]  # z 分量偏离 1
    l2 = deviation**2  # L2 penalty
    return l2

def fall_penalty(env):
    """计算跌倒惩罚，基于身体与地面的高度差"""
    base_pos = env.robot.data.root_pos_w  # (num_envs, 3)
    height = base_pos[:, 2]  # z 轴高度

    # 假设跌倒阈值为0.5米
    fall_threshold = 0.5
    # 给一个固定的惩罚值200
    penalty = torch.where(
        height < fall_threshold,
        torch.tensor(10.0, device=height.device, dtype=height.dtype),
        torch.tensor(0.0, device=height.device, dtype=height.dtype),
    )
    return penalty


def joint_pos_limits(env):
    """当关节位置触及软限位时施加平方惩罚。"""

    joint_pos = env.robot.data.joint_pos
    limits = getattr(env.robot.data, "soft_joint_pos_limits", None)
    if limits is None:
        limits = getattr(env.robot.data, "joint_pos_limits", None)
    if limits is None:
        return torch.zeros(joint_pos.shape[0], device=env.device, dtype=joint_pos.dtype)

    if limits.shape[-1] != 2:
        raise RuntimeError(f"不支持的关节限位张量形状：{limits.shape}")

    limits = limits.to(device=joint_pos.device, dtype=joint_pos.dtype)
    lower = limits[..., 0]
    upper = limits[..., 1]

    below = torch.relu(lower - joint_pos)
    above = torch.relu(joint_pos - upper)
    penalty = below.pow(2) + above.pow(2)
    return penalty.sum(dim=1)


def joint_torques_l2(env):
    """对关节力矩的 L2 范数进行惩罚。"""

    joint_pos = env.robot.data.joint_pos
    torque = None
    for attr in ("joint_torque", "applied_joint_torque", "joint_effort"):
        torque = getattr(env.robot.data, attr, None)
        if torque is not None:
            break
    if torque is None:
        return torch.zeros(joint_pos.shape[0], device=env.device, dtype=joint_pos.dtype)
    torque = torque.to(device=joint_pos.device, dtype=joint_pos.dtype)
    return torch.sum(torque.pow(2), dim=1)


def joint_acc_l2(env):
    """关节加速度平方惩罚，若数据缺失则通过差分估计。"""

    joint_vel = env.robot.data.joint_vel
    joint_acc = getattr(env.robot.data, "joint_acc", None)
    if joint_acc is None:
        prev_joint_vel = getattr(env, "_prev_joint_vel", None)
        if prev_joint_vel is None:
            env._prev_joint_vel = joint_vel.clone()
            return torch.zeros(joint_vel.shape[0], device=env.device, dtype=joint_vel.dtype)
        dt = _get_sim_dt(env)
        joint_acc = (joint_vel - prev_joint_vel) / dt
        env._prev_joint_vel = joint_vel.clone()
    else:
        joint_acc = joint_acc.to(device=joint_vel.device, dtype=joint_vel.dtype)
    return torch.sum(joint_acc.pow(2), dim=1)


def action_rate_l2(env):
    """动作变化率惩罚，限制策略震荡。"""

    actions = getattr(env, "actions", None)
    if actions is None:
        joint_pos = env.robot.data.joint_pos
        return torch.zeros(joint_pos.shape[0], device=env.device, dtype=joint_pos.dtype)

    prev_actions = getattr(env, "_prev_actions", None)
    env._prev_actions = actions.clone()
    if prev_actions is None:
        return torch.zeros(actions.shape[0], device=actions.device, dtype=actions.dtype)

    prev_actions = prev_actions.to(device=actions.device, dtype=actions.dtype)
    delta = actions - prev_actions
    return torch.sum(delta.pow(2), dim=1)


def lin_vel_z_l2(env):
    """线速度 Z 分量的平方惩罚，鼓励身体高度稳定。"""

    lin_vel = getattr(env.robot.data, "root_lin_vel_w", None)
    if lin_vel is None:
        lin_vel = env.robot.data.root_lin_vel_b
    lin_vel = lin_vel.to(device=env.device)
    vz = lin_vel[:, 2]
    return vz.pow(2)


def ang_vel_xy_l2(env):
    """角速度 XY 分量的平方惩罚，限制横滚和俯仰震荡。"""

    ang_vel = getattr(env.robot.data, "root_ang_vel_b", None)
    if ang_vel is None:
        ang_vel = env.robot.data.root_ang_vel_w
    ang_vel = ang_vel.to(device=env.device)
    return ang_vel[:, :2].pow(2).sum(dim=1)