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

def action_velocity_continuity(env):
    """动作速度连续性惩罚，鼓励action变化平滑。
                因为action表示关节目标位置，所以此处等价于限制关节位置指令的一阶导变化率。
    """

    actions = getattr(env, "actions", None)
    if actions is None:
        joint_pos = env.robot.data.joint_pos
        return torch.zeros(joint_pos.shape[0], device=env.device, dtype=joint_pos.dtype)

    # 取上一次的action和上上次的action
    prev_actions = getattr(env, "_prev_actions", None)
    prev_vel = getattr(env, "_prev_action_vel", None)
   
    # 当前动作速度（位置变化率）
    if prev_actions is not None:
        curr_vel = actions - prev_actions
    else:
        curr_vel = torch.zeros_like(actions)

    # 存储当前状态供下次调用
    env._prev_action_vel = curr_vel.clone()
    env._prev_actions = actions.clone()

    # 若还没有历史数据，则返回0
    if prev_vel is None or prev_actions is None:
        return torch.zeros(actions.shape[0], device=actions.device, dtype=actions.dtype)

    # 计算动作速度变化率（二阶差分）
    delta_vel = curr_vel - prev_vel

    # L2 范数惩罚
    return torch.sum(delta_vel.pow(2), dim=1)


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



def get_gait_phase_reward_(env):
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
    r_phase = torch.tanh(r_phase * 5.0)
    # 限制最大值为 0.4
    r_phase = torch.clamp(r_phase, max=0.4, min=-0.4)

    return r_phase


def get_gait_phase_reward(env):
    """
    基于 gait phase 的步态节奏奖励（仅考虑脚高度）。
    当 gait phase 为正半周 (sin>0) 时，左脚应高、右脚应低；
    当为负半周 (sin<0) 时，右脚应高、左脚应低。

    目标高度波动幅度为 ±0.2 m。
    """

    body_pos = env.robot.data.body_pos_w   # (num_envs, num_bodies, 3)
    l_id, r_id = env._l, env._r

    # 当前脚的世界坐标高度
    zL, zR = body_pos[:, l_id, 2], body_pos[:, r_id, 2]
    #print('zL:', zL[0].item(), ' zR:', zR[0].item())

    # 当前步态相位（假设随时间线性增加）
    phase = env.gait_phase
    phase_sin = torch.sin(phase)

    # 理想的脚高度曲线：sin(phase) 对应的目标高度
    # 左脚：在 sin>0 时高，右脚相反
    target_L = 0.2 * torch.clamp(phase_sin, min=0.0)   # 正半周抬高到 +0.2m
    target_R = 0.2 * torch.clamp(-phase_sin, min=0.0)  # 负半周抬高到 +0.2m

    # 实际脚高度与目标高度的偏差
    err_L = (zL - target_L).pow(2)
    err_R = (zR - target_R).pow(2)

    # 总误差
    total_err = err_L + err_R

    # 平滑奖励：误差越小奖励越高
    r_phase = torch.exp(-50.0 * total_err)   # 50 控制容忍度，可调 20–80 之间
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


def joint_sum_l2(env, joint_names):
    """
    鼓励指定关节的角度之和接近 0。

    参数:
        env: 仿真环境对象，包含 env.robot.data.joint_pos。
        joint_names: list[str]，指定需要约束的关节名列表。
            例如：
                ["left_shoulder_pitch", "right_shoulder_pitch", "torso_yaw"]

    返回:
        (num_envs,) 张量，表示每个环境的惩罚值。
    """
    joint_pos = env.robot.data.joint_pos  # (num_envs, num_joints)
    all_joint_names = env.robot.joint_names

    # 找出这些关节的索引
    indices = []
    for name in joint_names:
        try:
            indices.append(all_joint_names.index(name))
        except ValueError:
            raise ValueError(f"无法在 joint_names 中找到关节：{name}")

    if len(indices) == 0:
        return torch.zeros(joint_pos.shape[0], device=env.device, dtype=joint_pos.dtype)

    # 提取这些关节的角度
    selected = joint_pos[:, indices]  # (num_envs, len(indices))

    # 对每个环境计算这些角度的和
    sum_val = selected.sum(dim=1)

    # 目标是让 sum_val ≈ 0
    loss = sum_val.pow(2)

    return loss

#两个关节相等
def joint_equal_l2(env, joint_name_a, joint_name_b):
    """
    鼓励指定的两个关节角度相等。

    参数:
        env: 仿真环境对象，包含 env.robot.data.joint_pos。
        joint_name_a: str，第一个关节名。
        joint_name_b: str，第二个关节名。
    """
    joint_pos = env.robot.data.joint_pos
    all_joint_names = env.robot.joint_names

    try:
        idx_a = all_joint_names.index(joint_name_a)
        idx_b = all_joint_names.index(joint_name_b)
    except ValueError as e:
        raise ValueError(f"无法在 joint_names 中找到指定的关节：{e}")

    # 计算两个关节的偏差
    deviation = joint_pos[:, idx_a] - joint_pos[:, idx_b]
    return deviation.pow(2)

# 以下为command相关奖励函数
def command_lin_vel_tracking_reward(env):
    """鼓励机体线速度在水平面上跟随目标指令。"""

    if not hasattr(env, "commands"):
        return torch.zeros(env.robot.data.root_lin_vel_b.shape[0], device=env.device)

    base_lin_vel = env.robot.data.root_lin_vel_b
    target = env.commands[:, :2]

    tracking_error = base_lin_vel[:, :2] - target
    # 使用高斯型奖励，使误差越小说明越接近目标方向
    sigma_sq = 0.25  # (m/s)^2
    reward = torch.exp(-tracking_error.pow(2).sum(dim=1) / sigma_sq)
    return reward


def command_ang_vel_tracking_reward(env):
    """鼓励机体偏航角速度跟随目标指令。"""

    if not hasattr(env, "commands"):
        return torch.zeros(env.robot.data.root_ang_vel_b.shape[0], device=env.device)

    base_ang_vel = env.robot.data.root_ang_vel_b[:, 2]
    target = env.commands[:, 2]

    tracking_error = base_ang_vel - target
    sigma_sq = 0.5  # (rad/s)^2
    reward = torch.exp(-tracking_error.pow(2) / sigma_sq)
    return reward
    
    

def gait_phase_symmetry_reward(env, joint_pairs, alpha=10.0, beta=5.0, alt_limit=0.5):
    """
    改进版步态对称奖励：
    - 对称性：相反相位（0↔180, 90↔270）左右关节应相等；
    - 交替性：同一关节在相反相位差值大（但有上限）。

    参数:
        env: 环境对象，需包含 phase_key_angles。
        joint_pairs: list[[str,str]] 对称关节名对。
        alpha: 控制对称性惩罚强度。
        beta: 控制交替性强化强度。
        alt_limit: 交替差值的最大有效幅度（单位：弧度）。
    """
    if not hasattr(env, "phase_key_angles"):
        raise RuntimeError("env.phase_key_angles 未初始化。")
    if any(k not in env.phase_key_angles or env.phase_key_angles[k] is None for k in [0, 1, 2, 3]):
        N = getattr(env, "num_envs", 1)
        return torch.zeros(N, device=env.device)

    pos0 = env.phase_key_angles[0]
    pos1 = env.phase_key_angles[1]
    pos2 = env.phase_key_angles[2]
    pos3 = env.phase_key_angles[3]
    name_to_idx = {n: i for i, n in enumerate(env.robot.joint_names)}

    sym_errs, alt_diffs = [], []
    for left, right in joint_pairs:
        if left not in name_to_idx or right not in name_to_idx:
            raise ValueError(f"未找到关节：{left} 或 {right}")
        l_idx, r_idx = name_to_idx[left], name_to_idx[right]

        # --- 对称相位下左右关节应相等 ---
        diff_sym_1 = (pos0[:, l_idx] - pos2[:, r_idx]).pow(2)
        diff_sym_2 = (pos1[:, l_idx] - pos3[:, r_idx]).pow(2)
        sym_errs.append(diff_sym_1 + diff_sym_2)

        # --- 同一关节在相反相位间应差异较大 ---
        alt_L = torch.maximum(
            torch.abs(pos0[:, l_idx] - pos2[:, l_idx]),
            torch.abs(pos1[:, l_idx] - pos3[:, l_idx]),
        )
        alt_R = torch.maximum(
            torch.abs(pos0[:, r_idx] - pos2[:, r_idx]),
            torch.abs(pos1[:, r_idx] - pos3[:, r_idx]),
        )
        # 限幅并归一化到 [0,1]
        alt_val = torch.tanh(alt_L / alt_limit) * 0.5 + torch.tanh(alt_R / alt_limit) * 0.5
        alt_diffs.append(alt_val)

    sym_err = torch.stack(sym_errs, dim=1).mean(dim=1)
    alt_diff = torch.stack(alt_diffs, dim=1).mean(dim=1)

    reward_sym = torch.exp(-alpha * sym_err)
    reward_alt = torch.tanh(beta * alt_diff)
    reward = 0.5 * reward_sym + 0.5 * reward_alt
    return reward
    
def feet_air_time_biped(env, link_names, threshold: float = 0.5):
    """
         奖励双足交替迈步（单脚支撑）行为。

         参数:
        env: 仿真环境对象，需包含 env.scene.sensors['contact_sensor']。
        link_names: list[str]，指定脚部 link 名称列表（例如 ["left_foot", "right_foot"]）。
        threshold: float，最大奖励对应的腾空/接触持续时间阈值（秒）。

         返回:
        (num_envs,) 张量，表示每个环境的奖励值。
    """

    # 获取脚部对应的 ID
    contact_sensor = env.scene.sensors.get("contact_forces", None)
    if contact_sensor is None:
        raise ValueError("未在 env.scene.sensors 中找到 'contact_forces'。")

    all_body_names = env.scene.sensors["contact_forces"].body_names
    
    indices = []
    for name in link_names:
        try:
            indices.append(all_body_names.index(name))
        except ValueError:
            raise ValueError(f"无法在 rigid_body_names 中找到 link：{name}")

    if len(indices) == 0:
        return torch.zeros(env.num_envs, device=env.device)
        
    #print('all contact time:', contact_sensor.data.current_contact_time)
    #print('indices',indices)

    # 获取对应 link 的触地与腾空时间
    air_time = contact_sensor.data.current_air_time[:, indices]
    #print('air time:', air_time)
    contact_time = contact_sensor.data.current_contact_time[:, indices]
    #print('contact_time:', contact_time)
    in_contact = contact_time > 0.0
    #print('in_contact:', in_contact)

    # 当前“模式时间”：如果在地面则用接触时间，否则用腾空时间
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    #print('in_mode_time:', in_mode_time)

    # 单脚支撑状态（正好一只脚在地）
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    #print('single_stance:', single_stance)


    # 对单脚支撑状态给予奖励：取两脚的最小 in_mode_time（避免双脚状态极端不平衡）
    reward = torch.min(
        torch.where(single_stance.unsqueeze(-1), in_mode_time, torch.zeros_like(in_mode_time)),
        dim=1
    )[0]
    
    #print('reward:', reward)


    # 限制奖励上限
    reward = torch.clamp(reward, max=threshold)
    #print('reward after clamp:', reward)

    # 惩罚持续时间过长的状态（超过 1.5×threshold）
    excessive_time = torch.max(in_mode_time, dim=1)[0] > (threshold * 1.5)
    #print('excessive_time:', excessive_time)
    
    penalty = excessive_time.float() * -0.1
    #print('penalty:', penalty)
    
    reward = reward + penalty
    #print('final reward:', reward)

    return reward
