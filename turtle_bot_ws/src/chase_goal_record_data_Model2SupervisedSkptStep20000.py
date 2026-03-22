      
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

import torch
import torch.nn as nn


DEFAULT_TIMEOUT_SEC = 90.0


class DataRecorder:
    def __init__(self, filename="experiment_all_goals.csv", extra_fields: Optional[list] = None):
        self.filename = filename
        self.extra_fields = extra_fields if extra_fields is not None else []
        self.header = [
            "timestamp", "goal_name",
            "goal_x", "goal_y", "ux", "uy", "pos_x", "pos_y", "yaw", "vel_v", "vel_w"
        ] + self.extra_fields
        with open(self.filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.header)
        print(f"[Recorder] 连续数据记录已启动，文件: {self.filename}")

    def record(
        self,
        goal_name,
        goal_x,
        goal_y,
        ux,
        uy,
        pose,
        v,
        w,
        extra_data: Optional[Dict[str, Any]] = None,
    ):
        if pose is None:
            return
        x, y, yaw = pose
        extra_data = extra_data if extra_data is not None else {}
        extra_values = [extra_data.get(field, "") for field in self.extra_fields]
        with open(self.filename, mode="a", newline="") as f:
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
        self._last_scan = {
            "ranges": msg.ranges,
            "angle_min": msg.angle_min,
            "angle_increment": msg.angle_increment,
            "range_min": msg.range_min,
            "range_max": msg.range_max,
        }

    def get_pose(self):
        return self._last_pose

    def get_scan(self):
        return self._last_scan

    def send_velocity(self, v: float, w: float):
        msg = Twist()
        msg.linear.x = v
        msg.angular.z = w
        self._cmd_pub.publish(msg)

    def stop_robot(self):
        self.send_velocity(0.0, 0.0)


# ==================== 1. 复用原代码的网络定义（必须保留） ====================
class Actor(nn.Module):
    """监督策略网络（拆分输入分支：LiDAR+状态特征）"""

    def __init__(self, lidar_dim=36, state_dim=6, act_dim=2, hidden_dim=128, share_bb=None):
        super(Actor, self).__init__()

        self.share_bb = share_bb
        self.lidar_branch = nn.Sequential(
            nn.Linear(lidar_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        self.state_branch = nn.Sequential(
            nn.Linear(state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
        )

        self.fc_merge = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_dim // 2 + hidden_dim // 2, hidden_dim),
        )
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

    def predict_mean(self, x):
        mean, _ = self.forward(x)
        return mean


_SUP_ACTOR = None
_SUP_DEVICE = None
_SUP_STATE_DIM = 4
_SUP_LAST_ACTION = (0.0, 0.0)


def _reset_supervised_control_state():
    global _SUP_LAST_ACTION
    _SUP_LAST_ACTION = (0.0, 0.0)


def _safe_torch_load_checkpoint(torch_module, ckpt_path: str, device):
    try:
        return torch_module.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        return torch_module.load(ckpt_path, map_location=device)


def _get_actor_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    if "actor_state_dict" in checkpoint:
        return checkpoint["actor_state_dict"]
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    raise KeyError("未在checkpoint中找到 actor_state_dict/model_state_dict/state_dict")


def _infer_state_dim_from_state_dict(actor_state_dict: Dict[str, Any], fallback: int = 4) -> int:
    key = "state_branch.0.weight"
    if key in actor_state_dict and hasattr(actor_state_dict[key], "shape"):
        return int(actor_state_dict[key].shape[1])
    return fallback


def _load_supervised_actor(ckpt_path: str):
    global _SUP_ACTOR, _SUP_DEVICE, _SUP_STATE_DIM
    if _SUP_ACTOR is not None:
        return _SUP_ACTOR, _SUP_DEVICE, _SUP_STATE_DIM

    _SUP_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"ckpt文件不存在: {ckpt_path}")

    checkpoint = _safe_torch_load_checkpoint(torch, ckpt_path, _SUP_DEVICE)
    actor_state_dict = _get_actor_state_dict(checkpoint)
    _SUP_STATE_DIM = _infer_state_dim_from_state_dict(actor_state_dict, fallback=4)

    actor = Actor(lidar_dim=36, state_dim=_SUP_STATE_DIM, act_dim=2, hidden_dim=128).to(_SUP_DEVICE)
    actor.load_state_dict(actor_state_dict)
    actor.eval()

    _SUP_ACTOR = actor
    return _SUP_ACTOR, _SUP_DEVICE, _SUP_STATE_DIM


def control_policy_model_supervised_ckpt(
    x: float,
    y: float,
    yaw: float,
    gx: float,
    gy: float,
    scan: Dict[str, Any],
    ckpt_path: Optional[str] = None,
) -> Tuple[float, float]:
    import numpy as np  # pylint: disable=import-outside-toplevel

    global _SUP_LAST_ACTION

    if ckpt_path is None:
        ckpt_path = os.path.join(
            os.path.dirname(__file__), "Models", "model_2_supervised_ckpt_step_200000.pth"
        )

    actor, device, state_dim = _load_supervised_actor(ckpt_path)

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

    prev_v, prev_w = _SUP_LAST_ACTION
    max_v = 1.0
    max_w = 1.0
    prev_v_norm = np.clip(prev_v / max_v, 0.0, 1.0)
    prev_w_norm = np.clip(prev_w / max_w, -1.0, 1.0)

    if state_dim >= 6:
        state_obs = np.array(
            [ux, uy, prev_v_norm, prev_w_norm, prev_v_norm, prev_w_norm], dtype=np.float32
        )
    else:
        state_obs = np.array([ux, uy, prev_v_norm, prev_w_norm], dtype=np.float32)

    obs = np.concatenate([lidar_obs, state_obs], axis=0)
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        mean, _ = actor(obs_tensor)
        action_mean = mean.squeeze(0).cpu().numpy()

    raw_v_cmd = float(np.clip(action_mean[0] * max_v, 0.0, max_v))
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

    _SUP_LAST_ACTION = (v_cmd, w_cmd)
    return v_cmd, w_cmd


# 兼容旧调用名：外部仍可按 Model1 的函数名调用。
def control_policy_model_ppo_ckpt(
    x: float,
    y: float,
    yaw: float,
    gx: float,
    gy: float,
    scan: Dict[str, Any],
    ckpt_path: Optional[str] = None,
) -> Tuple[float, float]:
    return control_policy_model_supervised_ckpt(x, y, yaw, gx, gy, scan, ckpt_path)


def track_single_goal(
    goal_xy: Tuple[float, float],
    recorder: DataRecorder,
    goal_name: str = "unknown",
    reach_threshold_m: float = 0.25,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    control_rate_hz: float = 10.0,
    collision_fail_distance_m: float = 0.3,
    scene_data: Optional[Dict[str, Any]] = None,
) -> bool:
    if not rclpy.ok():
        rclpy.init()
    _reset_supervised_control_state()
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
            if pose is None or scan is None:
                continue

            ranges = scan.get("ranges", [])
            angle_min = float(scan.get("angle_min", -math.pi))
            angle_increment = float(scan.get("angle_increment", 0.0))
            rmin = float(scan.get("range_min", 0.0))
            rmax = float(scan.get("range_max", float("inf")))

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

            dx, dy = gx - x, gy - y
            angle_to_goal = math.atan2(dy, dx) - yaw
            angle_to_goal = math.atan2(math.sin(angle_to_goal), math.cos(angle_to_goal))

            ux = math.cos(angle_to_goal)
            uy = math.sin(angle_to_goal)

            v, w = control_policy_model_supervised_ckpt(x, y, yaw, gx, gy, scan)
            recorder.record(goal_name, gx, gy, ux, uy, pose, v, w, extra_data=scene_data)

            node.send_velocity(v, w)
            time.sleep(1.0 / control_rate_hz)

    except KeyboardInterrupt:
        pass
    finally:
        _reset_supervised_control_state()
        node.stop_robot()
        node.destroy_node()
    return False

    