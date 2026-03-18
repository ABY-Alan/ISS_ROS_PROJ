#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple, Optional, Dict, Any
import math
import time
import csv
import os
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


DEFAULT_TIMEOUT_SEC = 90.0

# --- 数据记录器类 ---

class DataRecorder:
    def __init__(self, filename="experiment_all_goals.csv", extra_fields: Optional[list] = None):
        # 记录器只在初始化时创建文件并写入表头
        self.filename = filename
        self.extra_fields = extra_fields if extra_fields is not None else []
        self.header = [
            "timestamp", "goal_name", # 新增 goal_name 用于区分
            "goal_x", "goal_y", "ux", "uy", "pos_x", "pos_y", "yaw", "vel_v", "vel_w"
        ] + self.extra_fields
        with open(self.filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.header)
        print(f"[Recorder] 连续数据记录已启动，文件: {self.filename}")

    def record(self, goal_name, goal_x, goal_y, ux, uy, pose, v, w, extra_data: Optional[Dict[str, Any]] = None):
        if pose is None: return
        x, y, yaw = pose
        extra_data = extra_data if extra_data is not None else {}
        extra_values = [extra_data.get(field, "") for field in self.extra_fields]
        with open(self.filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([time.time(), goal_name, goal_x, goal_y, ux, uy, x, y, yaw, v, w] + extra_values)
            
def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

class _GoalTracker(Node):
    def __init__(self):
        super().__init__("goal_tracker_min")
        self._odom_sub = self.create_subscription(Odometry, "/odom", self._on_odom, 10)
        self._last_pose: Optional[Tuple[float, float, float]] = None
        self._scan_sub = self.create_subscription(LaserScan, "/scan", self._on_scan, 10)
        self._last_scan: Optional[Dict[str, Any]] = None
        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
        self._last_pose = (p.x, p.y, yaw)

    def _on_scan(self, msg: LaserScan):
        # 兼容 control_policy 中的调用方式
        self._last_scan = {
            "ranges": msg.ranges,
            "angle_min": msg.angle_min,
            "angle_increment": msg.angle_increment,
            "range_min": msg.range_min,
            "range_max": msg.range_max
        }

    def get_pose(self): return self._last_pose
    def get_scan(self): return self._last_scan

    def send_velocity(self, v: float, w: float):
        msg = Twist()
        msg.linear.x = v
        msg.angular.z = w
        self._cmd_pub.publish(msg)

    def stop_robot(self):
        self.send_velocity(0.0, 0.0)


_PPO_ACTOR = None
_PPO_DEVICE = None
_PPO_LAST_ACTION = (0.0, 0.0)


def _reset_ppo_control_state():
    global _PPO_LAST_ACTION
    _PPO_LAST_ACTION = (0.0, 0.0)


def _safe_torch_load_checkpoint(torch_module, ckpt_path: str, device):
    # 新版本PyTorch建议显式设置weights_only=True；旧版本做参数兼容回退。
    try:
        return torch_module.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        return torch_module.load(ckpt_path, map_location=device)


def _load_ppo_actor(ckpt_path: str):
    """按需加载PPO Actor并缓存，避免每个控制周期重复加载。"""
    global _PPO_ACTOR, _PPO_DEVICE
    if _PPO_ACTOR is not None:
        return _PPO_ACTOR, _PPO_DEVICE

    import torch  # pylint: disable=import-outside-toplevel
    import torch.nn as nn  # pylint: disable=import-outside-toplevel

    class Actor(nn.Module):
        def __init__(self, lidar_dim=36, state_dim=6, act_dim=2, hidden_dim=128):
            super().__init__()
            self.lidar_branch = nn.Sequential(
                nn.Linear(lidar_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
            )
            self.state_branch = nn.Sequential(
                nn.Linear(state_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, hidden_dim // 2),
                nn.Tanh(),
            )
            self.fc_merge = nn.Linear(hidden_dim // 2 + hidden_dim // 2, hidden_dim)
            self.mean_layer = nn.Linear(hidden_dim, act_dim)
            self.log_std_layer = nn.Linear(hidden_dim, act_dim)
            self.log_std_min = -20
            self.log_std_max = 2

        def forward(self, x):
            lidar_feat = x[:, :36]
            state_feat = x[:, 36:]
            lidar_out = self.lidar_branch(lidar_feat)
            state_out = self.state_branch(state_feat)
            merge_feat = torch.cat([lidar_out, state_out], dim=1)
            merge_feat = torch.tanh(self.fc_merge(merge_feat))
            mean = self.mean_layer(merge_feat)
            log_std = self.log_std_layer(merge_feat)
            log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
            return mean, log_std

    _PPO_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = Actor(lidar_dim=36, state_dim=6, act_dim=2, hidden_dim=128).to(_PPO_DEVICE)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"ckpt文件不存在: {ckpt_path}")

    checkpoint = _safe_torch_load_checkpoint(torch, ckpt_path, _PPO_DEVICE)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    actor.eval()

    _PPO_ACTOR = actor
    return _PPO_ACTOR, _PPO_DEVICE


def control_policy_model_ppo_ckpt(
    x: float,
    y: float,
    yaw: float,
    gx: float,
    gy: float,
    scan: Dict[str, Any],
    ckpt_path: Optional[str] = None,
) -> Tuple[float, float]:
    """基于PPO ckpt推理输出速度指令。"""
    import numpy as np  # pylint: disable=import-outside-toplevel
    import torch  # pylint: disable=import-outside-toplevel
    global _PPO_LAST_ACTION

    if ckpt_path is None:
        ckpt_path = os.path.join(
            os.path.dirname(__file__), "Models", "ppo_ckpt_step_10000.pth"
        )

    actor, device = _load_ppo_actor(ckpt_path)

    ranges = np.asarray(scan.get("ranges", []), dtype=np.float32)
    if ranges.size == 0:
        ranges = np.ones(36, dtype=np.float32)

    rmin = float(scan.get("range_min", 0.0))
    rmax = float(scan.get("range_max", 3.5))
    valid_mask = np.isfinite(ranges) & (ranges >= rmin) & (ranges <= rmax)
    ranges = np.where(valid_mask, ranges, rmax)

    sample_idx = np.linspace(0, len(ranges) - 1, 36, dtype=int)
    lidar_obs = ranges[sample_idx]
    lidar_obs = np.clip(lidar_obs / max(rmax, 1e-6), 0.0, 1.0)

    dx = gx - x
    dy = gy - y
    angle_to_goal = math.atan2(dy, dx) - yaw
    angle_to_goal = math.atan2(math.sin(angle_to_goal), math.cos(angle_to_goal))
    ux = math.cos(angle_to_goal)
    uy = math.sin(angle_to_goal)

    # 6维状态: UserIntent(2) + BaseVel(2) + ActionHistory(2)
    prev_v, prev_w = _PPO_LAST_ACTION
    max_v = 0.5
    max_w = 1.0
    prev_v_norm = np.clip(prev_v / max_v, 0.0, 1.0)
    prev_w_norm = np.clip(prev_w / max_w, -1.0, 1.0)
    state_obs = np.array([ux, uy, prev_v_norm, prev_w_norm, prev_v_norm, prev_w_norm], dtype=np.float32)
    obs = np.concatenate([lidar_obs, state_obs], axis=0)
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        mean, _ = actor(obs_tensor)
        action_mean = mean.squeeze(0).cpu().numpy()

    raw_v_cmd = float(np.clip(action_mean[0] * 0.5, 0.0, 0.5))
    raw_w_cmd = float(np.clip(action_mean[1] * max_w, -max_w, max_w))

    front_sector = lidar_obs[14:22]
    front_clearance = float(np.min(front_sector)) if front_sector.size > 0 else 1.0
    clearance_scale = np.clip((front_clearance - 0.10) / 0.35, 0.0, 1.0)

    heading_w = float(np.clip(1.8 * angle_to_goal, -max_w, max_w))
    if abs(angle_to_goal) > 0.9:
        v_cmd = 0.0
        w_cmd = heading_w
    else:
        v_cmd = min(raw_v_cmd, max_v) * clearance_scale
        if abs(angle_to_goal) < 0.35 and front_clearance > 0.45:
            v_cmd = max(v_cmd, 0.08)
        w_cmd = float(np.clip(0.55 * raw_w_cmd + 0.45 * heading_w, -max_w, max_w))

    _PPO_LAST_ACTION = (v_cmd, w_cmd)
    return v_cmd, w_cmd

'''
def control_policy_model_ppo_blend(
    x: float,
    y: float,
    yaw: float,
    gx: float,
    gy: float,
    scan: Dict[str, Any],
) -> Tuple[float, float]:
    """模型优先控制：PPO输出 + 朝向纠偏 + 对称避障修正。"""
    model_v, model_w = control_policy_model_ppo_ckpt(x, y, yaw, gx, gy, scan)

    dx = gx - x
    dy = gy - y
    goal_yaw = math.atan2(dy, dx)
    angle_error = goal_yaw - yaw
    angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))

    ranges = scan.get("ranges", [])
    angle_min = float(scan.get("angle_min", -math.pi))
    angle_inc = float(scan.get("angle_increment", 0.0))
    rmin = float(scan.get("range_min", 0.0))
    rmax = float(scan.get("range_max", 3.5))

    front_vals = []
    left_vals = []
    right_vals = []
    for i, raw in enumerate(ranges):
        if raw is None or (not math.isfinite(raw)):
            continue
        dist = float(raw)
        if not (rmin <= dist <= rmax):
            continue

        ang = angle_min + i * angle_inc
        if abs(ang) <= 0.35:
            front_vals.append(dist)
        if 0.25 <= ang <= 1.10:
            left_vals.append(dist)
        if -1.10 <= ang <= -0.25:
            right_vals.append(dist)

    front_min = min(front_vals) if front_vals else rmax
    left_min = min(left_vals) if left_vals else rmax
    right_min = min(right_vals) if right_vals else rmax

    max_v = 0.22
    max_w = 1.0
    avoid_trigger = 0.55
    avoid_stop = 0.25

    heading_w = float(max(-max_w, min(max_w, 1.8 * angle_error)))

    # 左右扇区差值修正，抵消模型潜在单侧偏置。
    side_diff = left_min - right_min
    side_bias_w = float(max(-0.35, min(0.35, 0.9 * side_diff)))

    avoid_w = 0.0
    if front_min < avoid_trigger:
        turn_dir = 1.0 if left_min >= right_min else -1.0
        ratio = (avoid_trigger - front_min) / max(avoid_trigger - avoid_stop, 1e-6)
        ratio = max(0.0, min(1.0, ratio))
        avoid_w = turn_dir * (0.40 + 0.60 * ratio) * max_w

    # 模型为主，加入目标朝向与环境对称修正。
    if front_min < avoid_trigger:
        w = 0.55 * model_w + 0.20 * heading_w + 0.25 * avoid_w
    else:
        w = 0.70 * model_w + 0.25 * heading_w + 0.05 * side_bias_w
    w = float(max(-max_w, min(max_w, w)))

    if abs(angle_error) > 1.05:
        v = min(model_v, 0.06)
    else:
        clearance_scale = max(0.0, min(1.0, (front_min - avoid_stop) / (1.20 - avoid_stop)))
        v = min(model_v, max_v) * clearance_scale
        if front_min > 0.80 and abs(angle_error) < 0.25:
            v = max(v, 0.08)
    v = float(max(0.0, min(max_v, v)))

    return v, w
'''

def track_single_goal(
    goal_xy: Tuple[float, float],
    recorder: DataRecorder, # 新增参数
    goal_name: str = "unknown", 
    reach_threshold_m: float = 0.25,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    control_rate_hz: float = 10.0,
    collision_fail_distance_m: float = 0.3,
    scene_data: Optional[Dict[str, Any]] = None,
) -> bool:
    if not rclpy.ok(): rclpy.init()
    _reset_ppo_control_state()
    if scene_data is None:
        scene_data = {}
    scene_data.setdefault("collision_happened", 0)
    
    node = _GoalTracker()
    gx, gy = float(goal_xy[0]), float(goal_xy[1])
    start_time = time.time()
    collision_grace_period_sec = 1.0
    collision_confirm_count = 3
    front_half_angle_rad = 0.70
    consecutive_collision_hits = 0
    
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            
            if (time.time() - start_time) > timeout_sec:
                print(f"{goal_name} 超时")
                break

            pose = node.get_pose()
            scan = node.get_scan()
            if pose is None or scan is None: continue

            ranges = scan.get("ranges", [])
            angle_min = float(scan.get("angle_min", -math.pi))
            angle_increment = float(scan.get("angle_increment", 0.0))
            rmin = float(scan.get("range_min", 0.0))
            rmax = float(scan.get("range_max", float("inf")))

            # 仅在前向扇区检测碰撞，避免刚生成时个别侧向/异常近距射线导致误判
            front_collision_valid = []
            for idx, raw_range in enumerate(ranges):
                if raw_range is None or (not math.isfinite(raw_range)):
                    continue
                angle = angle_min + idx * angle_increment
                if abs(angle) > front_half_angle_rad:
                    continue
                dist = float(raw_range)
                if 0.0 < dist <= rmax:
                    front_collision_valid.append(dist)

            min_dist = min(front_collision_valid) if front_collision_valid else float("inf")
            effective_collision_threshold = max(collision_fail_distance_m, rmin)

            if (time.time() - start_time) >= collision_grace_period_sec and min_dist <= effective_collision_threshold:
                consecutive_collision_hits += 1
            else:
                consecutive_collision_hits = 0

            if consecutive_collision_hits >= collision_confirm_count:
                node.stop_robot()
                scene_data["collision_happened"] = 1
                # 记录碰撞时刻，便于后续按行筛选碰撞样本
                x, y, yaw = pose
                dx, dy = gx - x, gy - y
                angle_to_goal = math.atan2(dy, dx) - yaw
                angle_to_goal = math.atan2(math.sin(angle_to_goal), math.cos(angle_to_goal))
                ux = math.cos(angle_to_goal)
                uy = math.sin(angle_to_goal)
                recorder.record(goal_name, gx, gy, ux, uy, pose, 0.0, 0.0, extra_data=scene_data)
                print(
                    f"失败: {goal_name} 检测到碰撞"
                    f"（min_scan={min_dist:.3f}m, threshold={effective_collision_threshold:.3f}m）"
                )
                return False

            x, y, yaw = pose
            dist = math.hypot(gx - x, gy - y)

            if dist <= reach_threshold_m:
                node.stop_robot()
                print(f"到达: {goal_name}")
                return True

            # 计算方向
            dx, dy = gx - x, gy - y
            angle_to_goal = math.atan2(dy, dx) - yaw
            angle_to_goal = math.atan2(math.sin(angle_to_goal), math.cos(angle_to_goal))
            
            # 模拟用户意图 (方向向量)
            ux = math.cos(angle_to_goal)
            uy = math.sin(angle_to_goal)

            # 使用带模型控制策略
            # v, w = control_policy_model_ppo_blend(x, y, yaw, gx, gy, scan)
            v, w = control_policy_model_ppo_ckpt(x, y, yaw, gx, gy, scan)
            
            # --- 关键：使用传入的 recorder 记录数据，带上 goal_name ---
            recorder.record(goal_name, gx, gy, ux, uy, pose, v, w, extra_data=scene_data)

            node.send_velocity(v, w)
            time.sleep(1.0 / control_rate_hz)

    except KeyboardInterrupt: pass
    finally:
        _reset_ppo_control_state()
        node.stop_robot()
        node.destroy_node()
    return False