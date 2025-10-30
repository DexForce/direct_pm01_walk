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



def get_gait_phase_reward(env):
    """
    基于 gait phase 的步态节奏奖励（无传感器版本）。
    当 gait phase 要求左脚摆动时，左脚应高、右脚应低；反之亦然。
    """

    # 获取脚部的位姿数据
    body_pos = env.robot.data.body_pos_w          # (num_envs, num_bodies, 3)
    body_vel = env.robot.data.body_vel_w          # (num_envs, num_bodies, 6)

    # 脚的索引（假设在 env.__init__ 里已缓存）
    l_id, r_id = env._l, env._r

    # 世界坐标下的 z 坐标和竖直速度
    zL, zR = body_pos[:, l_id, 2], body_pos[:, r_id, 2]
    vzL, vzR = body_vel[:, l_id, 2], body_vel[:, r_id, 2]

    # 步态相位：sin(phase)>0 时希望左脚摆动、右脚支撑；反之亦然
    phase = env.gait_phase
    left_should_swing = torch.sin(phase) > 0

    # “摆动度” = 高度差 + 竖直速度差
    # 越大表示越像“在摆动”
    swing_score_L = zL + 0.5 * vzL
    swing_score_R = zR + 0.5 * vzR

    # 奖励：期望的脚摆动得越高越好，另一只脚越低越好
    r_phase = torch.where(
        left_should_swing,
        swing_score_L - swing_score_R,
        swing_score_R - swing_score_L,
    )

    # 归一化与截断，避免极端大值
    r_phase = torch.tanh(r_phase * 3.0)
    # 限制最大值为 0.5
    r_phase = torch.clamp(r_phase, max=0.8) # tanh(x)=0.8时 x≈1.1，r_phase 最大约为1.1/3=0.37，左右脚“摆动度”差约0.37，高度差越0.2米。

    return r_phase


def joint_deviation_l1(env, joint_names=None):
    """指定关节的偏离默认角度的 L1 惩罚。
    
    参数:
        env: 环境对象，需包含 env.robot.data.joint_pos / default_joint_pos。
        joint_names: list[str] 或 None，指定要计算的关节名。
                      若为 None，则默认使用全部关节。
    返回:
        (num_envs,) 张量，表示每个环境的总偏差。
    """

    joint_pos = env.robot.data.joint_pos
    default_joint_pos = env.robot.data.default_joint_pos
    all_joint_names = env.robot.joint_names  # list[str]

    if joint_names is not None:
        # 找出指定关节的索引
        idx = [all_joint_names.index(name) for name in joint_names if name in all_joint_names]
        if len(idx) == 0:
            raise ValueError(f"未找到任何匹配的关节名：{joint_names}")
        joint_pos = joint_pos[:, idx]
        default_joint_pos = default_joint_pos[:, idx]

    # 计算 L1 偏离
    deviation = torch.abs(joint_pos - default_joint_pos)
    return torch.sum(deviation, dim=1)

def joint_symmetry_l2(env, joint_pairs):
    """
    鼓励指定成对关节相对于默认位置保持对称。

    参数:
        env: 仿真环境对象，包含 env.robot.data.joint_pos / default_joint_pos。
        joint_pairs: list[list[str]]，每个元素是长度为 2 的关节名对。
            例如：
                [
                    ["left_hip_yaw", "right_hip_yaw"],
                    ["left_knee", "right_knee"]
                ]

    返回:
        (num_envs,) 张量，表示每个环境的总对称性惩罚。
    """

    joint_pos = env.robot.data.joint_pos
    default_joint_pos = env.robot.data.default_joint_pos
    all_joint_names = env.robot.joint_names

    total_loss = torch.zeros(joint_pos.shape[0], device=env.device, dtype=joint_pos.dtype)

    for left_name, right_name in joint_pairs:
        # 找到两侧关节索引
        try:
            l_idx = all_joint_names.index(left_name)
            r_idx = all_joint_names.index(right_name)
        except ValueError as e:
            raise ValueError(f"无法在 joint_names 中找到指定的关节：{e}")

        # 各自相对于默认位置的偏移
        l_dev = joint_pos[:, l_idx] - default_joint_pos[:, l_idx]
        r_dev = joint_pos[:, r_idx] - default_joint_pos[:, r_idx]

        # 理想对称： l_dev ≈ -r_dev
        symmetry_error = (l_dev + r_dev).pow(2)
        total_loss += symmetry_error

    return total_loss