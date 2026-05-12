#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple, Optional, Dict, Any
import csv
import math
import os
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import torch

from PPO_ckpt_test import (
    CNNGRUSFTPolicy,
    CNNLSTMSFTPolicy,
    FCLSTMSFTPolicy,
    SimpleFCSFTPolicy,
)


DEFAULT_TIMEOUT_SEC = 90.0

MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "ppo_mixed_simple_fc_sft_epoch_020": {
        "policy_cls": SimpleFCSFTPolicy,
        "checkpoint": "ppo_mixed_simple_fc_sft_epoch_020.pth",
        "output_tag": "ppo_mixed_simple_fc_sft_epoch_020",
    },
    "ppo_mixed_cnn_lstm_sft_nodoor_epoch_020": {
        "policy_cls": CNNLSTMSFTPolicy,
        "checkpoint": "ppo_mixed_cnn_lstm_sft_nodoor_epoch_020.pth",
        "output_tag": "ppo_mixed_cnn_lstm_sft_nodoor_epoch_020",
    },
    "ppo_mixed_cnn_lstm_sft_epoch_020": {
        "policy_cls": CNNLSTMSFTPolicy,
        "checkpoint": "ppo_mixed_cnn_lstm_sft_epoch_020.pth",
        "output_tag": "ppo_mixed_cnn_lstm_sft_epoch_020",
    },
    "ppo_mixed_cnn_gru_sft_epoch_020": {
        "policy_cls": CNNGRUSFTPolicy,
        "checkpoint": "ppo_mixed_cnn_gru_sft_epoch_020.pth",
        "output_tag": "ppo_mixed_cnn_gru_sft_epoch_020",
    },
    "ppo_mixed_fc_lstm_sft_epoch_020": {
        "policy_cls": FCLSTMSFTPolicy,
        "checkpoint": "ppo_mixed_fc_lstm_sft_epoch_020.pth",
        "output_tag": "ppo_mixed_fc_lstm_sft_epoch_020",
    },
}

_MODEL_CACHE: Dict[str, torch.nn.Module] = {}
_MODEL_DEVICE: Optional[torch.device] = None
_LAST_ACTIONS: Dict[str, Tuple[float, float]] = {
    model_name: (0.0, 0.0) for model_name in MODEL_SPECS
}


class DataRecorder:
    def __init__(self, filename="experiment_all_goals.csv", extra_fields: Optional[list] = None):
        self.filename = filename
        self.extra_fields = extra_fields if extra_fields is not None else []
        self.header = [
            "timestamp", "goal_name",
            "goal_x", "goal_y", "ux", "uy", "pos_x", "pos_y", "yaw", "vel_v", "vel_w"
        ] + self.extra_fields
        with open(self.filename, mode="w", newline="") as file:
            writer = csv.writer(file)
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
        with open(self.filename, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([time.time(), goal_name, goal_x, goal_y, ux, uy, x, y, yaw, v, w] + extra_values)


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class _GoalTracker(Node):
    def __init__(self):
        super().__init__("goal_tracker_mixed_five_models")
        self._odom_sub = self.create_subscription(Odometry, "/odom", self._on_odom, 10)
        self._last_pose: Optional[Tuple[float, float, float]] = None
        self._scan_sub = self.create_subscription(LaserScan, "/scan", self._on_scan, 10)
        self._last_scan: Optional[Dict[str, Any]] = None
        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

    def _on_odom(self, msg: Odometry):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        yaw = quat_to_yaw(orientation.x, orientation.y, orientation.z, orientation.w)
        self._last_pose = (position.x, position.y, yaw)

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


def get_supported_model_names() -> Tuple[str, ...]:
    return tuple(MODEL_SPECS.keys())


def get_output_tag(model_name: str) -> str:
    _ensure_supported_model(model_name)
    return str(MODEL_SPECS[model_name]["output_tag"])


def _ensure_supported_model(model_name: str) -> None:
    if model_name not in MODEL_SPECS:
        supported = ", ".join(get_supported_model_names())
        raise ValueError(f"不支持的模型名: {model_name}。可选值: {supported}")


def _reset_control_state(model_name: str):
    _LAST_ACTIONS[model_name] = (0.0, 0.0)


def _safe_torch_load_checkpoint(torch_module, ckpt_path: str, device: torch.device):
    try:
        return torch_module.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        return torch_module.load(ckpt_path, map_location=device)


def _get_policy_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    if "policy_state_dict" in checkpoint:
        return checkpoint["policy_state_dict"]
    if "actor_state_dict" in checkpoint:
        return checkpoint["actor_state_dict"]
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    raise KeyError("未在checkpoint中找到 policy_state_dict/actor_state_dict/model_state_dict/state_dict")


def _get_device() -> torch.device:
    global _MODEL_DEVICE
    if _MODEL_DEVICE is None:
        _MODEL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _MODEL_DEVICE


def _load_policy(model_name: str) -> Tuple[torch.nn.Module, torch.device]:
    _ensure_supported_model(model_name)
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name], _get_device()

    spec = MODEL_SPECS[model_name]
    ckpt_path = os.path.join(os.path.dirname(__file__), "Models", spec["checkpoint"])
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"ckpt文件不存在: {ckpt_path}")

    device = _get_device()
    checkpoint = _safe_torch_load_checkpoint(torch, ckpt_path, device)
    policy_state_dict = _get_policy_state_dict(checkpoint)

    policy = spec["policy_cls"]().to(device)
    policy.load_state_dict(policy_state_dict)
    policy.eval()

    _MODEL_CACHE[model_name] = policy
    return policy, device


def control_policy_model_ppo_mixed(
    model_name: str,
    x: float,
    y: float,
    yaw: float,
    gx: float,
    gy: float,
    scan: Dict[str, Any],
) -> Tuple[float, float]:
    policy, device = _load_policy(model_name)

    ranges = np.asarray(scan.get("ranges", []), dtype=np.float32)
    if ranges.size == 0:
        ranges = np.ones(360, dtype=np.float32)

    rmin = float(scan.get("range_min", 0.0))
    rmax = float(scan.get("range_max", 3.5))
    valid_mask = np.isfinite(ranges) & (ranges >= rmin) & (ranges <= rmax)
    ranges = np.where(valid_mask, ranges, rmax)
    lidar_obs = np.clip(ranges / max(rmax, 1e-6), 0.0, 1.0)

    dx = gx - x
    dy = gy - y
    angle_to_goal = math.atan2(dy, dx) - yaw
    angle_to_goal = math.atan2(math.sin(angle_to_goal), math.cos(angle_to_goal))
    ux = math.cos(angle_to_goal)
    uy = math.sin(angle_to_goal)

    prev_v, prev_w = _LAST_ACTIONS[model_name]
    max_v = 1.0
    max_w = 1.0
    prev_v_norm = np.clip(prev_v / max_v, 0.0, 1.0)
    prev_w_norm = np.clip(prev_w / max_w, -1.0, 1.0)
    state_obs = np.array([ux, uy, prev_v_norm, prev_w_norm], dtype=np.float32)

    obs = np.concatenate([lidar_obs, state_obs], axis=0)
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        if hasattr(policy, "predict_mean"):
            action_mean_tensor = policy.predict_mean(obs_tensor)
        else:
            action_mean_tensor, _ = policy(obs_tensor)
        action_mean = action_mean_tensor.squeeze(0).cpu().numpy()

    raw_v_cmd = float(np.clip(action_mean[0] * max_v, 0.0, max_v))
    raw_w_cmd = float(np.clip(action_mean[1] * max_w, -max_w, max_w))

    front_sector = lidar_obs[150:210]
    front_clearance = float(np.min(front_sector)) if front_sector.size > 0 else 1.0
    clearance_scale = float(np.clip((front_clearance - 0.10) / 0.35, 0.0, 1.0))

    heading_w = float(np.clip(1.8 * angle_to_goal, -max_w, max_w))
    if abs(angle_to_goal) > 0.9:
        v_cmd = 0.0
        w_cmd = heading_w
    else:
        v_cmd = min(raw_v_cmd, max_v) * clearance_scale
        if abs(angle_to_goal) < 0.35 and front_clearance > 0.45:
            v_cmd = max(v_cmd, 0.08)
        w_cmd = float(np.clip(0.55 * raw_w_cmd + 0.45 * heading_w, -max_w, max_w))

    _LAST_ACTIONS[model_name] = (v_cmd, w_cmd)
    return v_cmd, w_cmd


def track_single_goal(
    model_name: str,
    goal_xy: Tuple[float, float],
    recorder: DataRecorder,
    goal_name: str = "unknown",
    reach_threshold_m: float = 0.25,
    timeout_sec: float = DEFAULT_TIMEOUT_SEC,
    control_rate_hz: float = 10.0,
    collision_fail_distance_m: float = 0.3,
    scene_data: Optional[Dict[str, Any]] = None,
) -> bool:
    _ensure_supported_model(model_name)
    if not rclpy.ok():
        rclpy.init()
    _reset_control_state(model_name)
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
            for index, raw_range in enumerate(ranges):
                if raw_range is None or (not math.isfinite(raw_range)):
                    continue
                angle = angle_min + index * angle_increment
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

            v, w = control_policy_model_ppo_mixed(model_name, x, y, yaw, gx, gy, scan)
            recorder.record(goal_name, gx, gy, ux, uy, pose, v, w, extra_data=scene_data)

            node.send_velocity(v, w)
            time.sleep(1.0 / control_rate_hz)

    except KeyboardInterrupt:
        pass
    finally:
        _reset_control_state(model_name)
        node.stop_robot()
        node.destroy_node()
    return False