#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple, Optional, Dict, Any
import math
import time
import csv
import random
import os
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

# --- 数据记录器类 ---

class DataRecorder:
    def __init__(self, filename="experiment_all_goals.csv"):
        # 记录器只在初始化时创建文件并写入表头
        self.filename = filename
        self.header = [
            "timestamp", "goal_name", # 新增 goal_name 用于区分
            "goal_x", "goal_y", "ux", "uy", "pos_x", "pos_y", "yaw", "vel_v", "vel_w"
        ]
        with open(self.filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.header)
        print(f"[Recorder] 连续数据记录已启动，文件: {self.filename}")

    def record(self, goal_name, goal_x, goal_y, ux, uy, pose, v, w):
        if pose is None: return
        x, y, yaw = pose
        with open(self.filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([time.time(), goal_name, goal_x, goal_y, ux, uy, x, y, yaw, v, w])
            
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

# ===== 保持不变的控制策略函数 =====
def control_policy_no_model_with_noise(
    x: float,
    y: float,
    yaw: float,
    gx: float,
    gy: float,
    scan: Dict[str, Any],
) -> Tuple[float, float]:
    dx = gx - x
    dy = gy - y
    goal_yaw = math.atan2(dy, dx)
    angle_error = goal_yaw - yaw
    angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))

    k_w = 2.0
    max_w = 1.5
    w = max(-max_w, min(max_w, k_w * angle_error))

    ranges = scan["ranges"]
    rmin = scan["range_min"]
    rmax = scan["range_max"]

    valid = [r for r in ranges if (r is not None) and math.isfinite(r) and (rmin <= r <= rmax)]
    avg_dist = (sum(valid) / len(valid)) if valid else rmax

    stop_dist = 0.35
    full_dist = 1.20
    v_max = 0.22
    v_min = 0.0
    angle_threshold = 0.35

    if abs(angle_error) > angle_threshold:
        v = 0.0
    elif avg_dist <= stop_dist:
        v = 0.0
    elif avg_dist >= full_dist:
        v = v_max
    else:
        ratio = (avg_dist - stop_dist) / (full_dist - stop_dist)
        v = v_min + ratio * (v_max - v_min)

    v_noise_std = 0.5
    w_noise_std = 0.3
    v += random.gauss(0.0, v_noise_std)
    w += random.gauss(0.0, w_noise_std)

    v = max(0.0, min(v_max, v))
    w = max(-max_w, min(max_w, w))

    return float(v), float(w)


_PPO_ACTOR = None
_PPO_DEVICE = None
_PPO_LAST_ACTION = (0.0, 0.0)


def _reset_ppo_control_state():
    global _PPO_LAST_ACTION
    _PPO_LAST_ACTION = (0.0, 0.0)


def _load_ppo_actor(ckpt_path: str):
    """按需加载PPO Actor并缓存，避免每个控制周期重复加载。"""
    global _PPO_ACTOR, _PPO_DEVICE
    if _PPO_ACTOR is not None:
        return _PPO_ACTOR, _PPO_DEVICE

    import numpy as np  # pylint: disable=import-outside-toplevel
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

    checkpoint = torch.load(ckpt_path, map_location=_PPO_DEVICE)
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
            os.path.dirname(__file__), "Models", "model_1_ppo_ckpt_step_10000.pth"
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
    max_v = 0.22
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


def track_single_goal(
    goal_xy: Tuple[float, float],
    recorder: DataRecorder, # 新增参数
    goal_name: str = "unknown", 
    reach_threshold_m: float = 0.25,
    timeout_sec: float = 60.0,
    control_rate_hz: float = 10.0,
) -> bool:
    if not rclpy.ok(): rclpy.init()
    _reset_ppo_control_state()
    
    node = _GoalTracker()
    gx, gy = float(goal_xy[0]), float(goal_xy[1])
    start_time = time.time()
    
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            
            if (time.time() - start_time) > timeout_sec:
                print(f"{goal_name} 超时")
                break

            pose = node.get_pose()
            scan = node.get_scan()
            if pose is None or scan is None: continue

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

            # 调用你的算法
            # v, w = control_policy_no_model_with_noise(x, y, yaw, gx, gy, scan)
            v, w = control_policy_model_ppo_ckpt(x, y, yaw, gx, gy, scan)
            
            # --- 关键：使用传入的 recorder 记录数据，带上 goal_name ---
            recorder.record(goal_name, gx, gy, ux, uy, pose, v, w)

            node.send_velocity(v, w)
            time.sleep(1.0 / control_rate_hz)

    except KeyboardInterrupt: pass
    finally:
        _reset_ppo_control_state()
        node.stop_robot()
        node.destroy_node()
    return False