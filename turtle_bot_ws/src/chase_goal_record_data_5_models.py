#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple, Optional, Dict, Any
import os
import csv
import time
import math

import numpy as np
import torch
import torch.nn as nn

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


# ============================================================
# Data Recorder
# ============================================================

class DataRecorder:
    def __init__(self, filename="experiment.csv", extra_fields: Optional[list] = None):
        self.filename = filename
        self.extra_fields = extra_fields or []

        self.header = [
            "timestamp",
            "goal_name",
            "goal_x",
            "goal_y",
            "ux",
            "uy",
            "pos_x",
            "pos_y",
            "yaw",
            "vel_v",
            "vel_w",
            "dist_to_goal",
            "angle_to_goal",
            "min_front_dist",
            "status",
        ] + self.extra_fields

        output_dir = os.path.dirname(filename)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(self.filename, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.header)

        print(f"[Recorder] file = {self.filename}")

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
        dist_to_goal="",
        angle_to_goal="",
        min_front_dist="",
        status="running",
        extra_data: Optional[Dict[str, Any]] = None,
    ):
        if pose is None:
            return

        x, y, yaw = pose
        extra_data = extra_data or {}
        extra_values = [extra_data.get(k, "") for k in self.extra_fields]

        row = [
            time.time(),
            goal_name,
            goal_x,
            goal_y,
            ux,
            uy,
            x,
            y,
            yaw,
            v,
            w,
            dist_to_goal,
            angle_to_goal,
            min_front_dist,
            status,
        ] + extra_values

        with open(self.filename, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)


# ============================================================
# ROS Utils
# ============================================================

def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


class GoalTracker(Node):
    def __init__(self):
        super().__init__("goal_tracker_5_models")

        self.pose: Optional[Tuple[float, float, float]] = None
        self.scan: Optional[Dict[str, Any]] = None

        self.create_subscription(
            Odometry,
            "/odom",
            self._on_odom,
            10,
        )

        self.create_subscription(
            LaserScan,
            "/scan",
            self._on_scan,
            10,
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            "/cmd_vel",
            10,
        )

    def _on_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        yaw = quat_to_yaw(q.x, q.y, q.z, q.w)
        self.pose = (p.x, p.y, yaw)

    def _on_scan(self, msg: LaserScan):
        self.scan = {
            "ranges": msg.ranges,
            "angle_min": msg.angle_min,
            "angle_increment": msg.angle_increment,
            "range_min": msg.range_min,
            "range_max": msg.range_max,
        }

    def send_velocity(self, v: float, w: float):
        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.cmd_pub.publish(msg)

    def stop_robot(self):
        self.send_velocity(0.0, 0.0)


# ============================================================
# Policy Networks
# ============================================================

class BasePolicy(nn.Module):
    """
    注意：
    forward 的参数名统一为 x，避免 Pylance 报：
    方法 forward 以不兼容的方式替代类 BasePolicy。
    """

    def __init__(self, obs_dim=364, act_dim=2):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def forward(self, x):
        raise NotImplementedError


class SimpleFCSFTPolicy(BasePolicy):
    """
    checkpoint key 对应：
    lidar_branch.*
    state_branch.*
    fc_merge.*
    mean_layer.*
    log_std_layer.*
    """

    def __init__(
        self,
        obs_dim=364,
        act_dim=2,
        lidar_dim=360,
        state_dim=4,
        hidden_dim=128,
    ):
        super().__init__(obs_dim=obs_dim, act_dim=act_dim)

        self.lidar_dim = int(lidar_dim)
        self.state_dim = int(state_dim)

        self.lidar_branch = nn.Sequential(
            nn.Linear(self.lidar_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        self.state_branch = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
        )

        self.fc_merge = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.mean_layer = nn.Linear(hidden_dim, act_dim)
        self.log_std_layer = nn.Linear(hidden_dim, act_dim)

        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, x):
        lidar_feat = x[:, :self.lidar_dim]
        state_feat = x[:, self.lidar_dim:self.lidar_dim + self.state_dim]

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


class CNNEncoder360(nn.Module):
    """
    必须保持这个内部名字：
        self.encoder

    因为 checkpoint 里的 key 是：
        cnn_body.encoder.0.weight
        cnn_body.encoder.0.bias
        cnn_body.encoder.2.weight
        cnn_body.encoder.2.bias
        cnn_body.encoder.6.weight
        cnn_body.encoder.6.bias
    """

    def __init__(self, out_dim=512):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=0),
            nn.ReLU(inplace=False),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=0),
            nn.ReLU(inplace=False),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(32 * 16, out_dim),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.encoder(x)


class CNNLSTMSFTPolicy(BasePolicy):
    """
    checkpoint key 对应：
        cnn_body.encoder.*
        lstm.*
        head.*
        mean_layer.*
        log_std_layer.*
    """

    def __init__(
        self,
        obs_dim=364,
        act_dim=2,
        lidar_dim=360,
        state_dim=4,
        cnn_vec_dim=512,
        lstm_hidden=128,
        head_hidden=128,
    ):
        super().__init__(obs_dim=obs_dim, act_dim=act_dim)

        self.lidar_dim = int(lidar_dim)
        self.state_dim = int(state_dim)

        # 名字必须是 cnn_body，不能改成 cnn
        self.cnn_body = CNNEncoder360(out_dim=cnn_vec_dim)

        self.lstm = nn.LSTM(
            input_size=cnn_vec_dim + self.state_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, head_hidden),
            nn.Tanh(),
        )

        self.mean_layer = nn.Linear(head_hidden, act_dim)
        self.log_std_layer = nn.Linear(head_hidden, act_dim)

        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, x):
        lidar = x[:, :self.lidar_dim]
        state = x[:, self.lidar_dim:self.lidar_dim + self.state_dim]

        z = self.cnn_body(lidar.unsqueeze(1))
        fused = torch.cat([z, state], dim=1).unsqueeze(1)

        h, _ = self.lstm(fused)
        h = h.squeeze(1)
        h = torch.clamp(h, -10.0, 10.0)

        feat = self.head(h)

        mean = self.mean_layer(feat)
        mean = torch.clamp(mean, -10.0, 10.0)

        log_std = self.log_std_layer(feat)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def predict_mean(self, x):
        mean, _ = self.forward(x)
        return mean


class CNNGRUSFTPolicy(BasePolicy):
    """
    checkpoint key 对应：
        cnn_body.encoder.*
        gru.*
        head.*
        mean_layer.*
        log_std_layer.*
    """

    def __init__(
        self,
        obs_dim=364,
        act_dim=2,
        lidar_dim=360,
        state_dim=4,
        cnn_vec_dim=512,
        rnn_hidden=128,
        head_hidden=128,
    ):
        super().__init__(obs_dim=obs_dim, act_dim=act_dim)

        self.lidar_dim = int(lidar_dim)
        self.state_dim = int(state_dim)

        # 名字必须是 cnn_body，不能改成 cnn
        self.cnn_body = CNNEncoder360(out_dim=cnn_vec_dim)

        self.gru = nn.GRU(
            input_size=cnn_vec_dim + self.state_dim,
            hidden_size=rnn_hidden,
            num_layers=1,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.Linear(rnn_hidden, head_hidden),
            nn.Tanh(),
        )

        self.mean_layer = nn.Linear(head_hidden, act_dim)
        self.log_std_layer = nn.Linear(head_hidden, act_dim)

        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, x):
        lidar = x[:, :self.lidar_dim]
        state = x[:, self.lidar_dim:self.lidar_dim + self.state_dim]

        z = self.cnn_body(lidar.unsqueeze(1))
        fused = torch.cat([z, state], dim=1).unsqueeze(1)

        h, _ = self.gru(fused)
        h = h.squeeze(1)
        h = torch.clamp(h, -10.0, 10.0)

        feat = self.head(h)

        mean = self.mean_layer(feat)
        mean = torch.clamp(mean, -10.0, 10.0)

        log_std = self.log_std_layer(feat)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def predict_mean(self, x):
        mean, _ = self.forward(x)
        return mean


class FCLSTMSFTPolicy(BasePolicy):
    """
    checkpoint key 对应：
        lidar_mlp.*
        lstm.*
        head.*
        mean_layer.*
        log_std_layer.*
    """

    def __init__(
        self,
        obs_dim=364,
        act_dim=2,
        lidar_dim=360,
        state_dim=4,
        lidar_emb_dim=128,
        lstm_hidden=128,
        head_hidden=128,
    ):
        super().__init__(obs_dim=obs_dim, act_dim=act_dim)

        self.lidar_dim = int(lidar_dim)
        self.state_dim = int(state_dim)

        self.lidar_mlp = nn.Sequential(
            nn.Linear(self.lidar_dim, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, lidar_emb_dim),
            nn.ReLU(inplace=False),
        )

        self.lstm = nn.LSTM(
            input_size=lidar_emb_dim + self.state_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, head_hidden),
            nn.Tanh(),
        )

        self.mean_layer = nn.Linear(head_hidden, act_dim)
        self.log_std_layer = nn.Linear(head_hidden, act_dim)

        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, x):
        lidar = x[:, :self.lidar_dim]
        state = x[:, self.lidar_dim:self.lidar_dim + self.state_dim]

        z = self.lidar_mlp(lidar)
        fused = torch.cat([z, state], dim=1).unsqueeze(1)

        h, _ = self.lstm(fused)
        h = h.squeeze(1)
        h = torch.clamp(h, -10.0, 10.0)

        feat = self.head(h)

        mean = self.mean_layer(feat)
        mean = torch.clamp(mean, -10.0, 10.0)

        log_std = self.log_std_layer(feat)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std

    def predict_mean(self, x):
        mean, _ = self.forward(x)
        return mean


# ============================================================
# Model Loading
# ============================================================

MODEL_CONFIGS = {
    "simple_fc_sft": {
        "class": SimpleFCSFTPolicy,
        "ckpt": "best_actor_simple_fc_sft.pth",
    },
    "cnn_lstm_sft_nodoor": {
        "class": CNNLSTMSFTPolicy,
        "ckpt": "best_actor_cnn_lstm_sft_nodoor.pth",
    },
    "cnn_lstm_sft": {
        "class": CNNLSTMSFTPolicy,
        "ckpt": "best_actor_cnn_lstm_sft.pth",
    },
    "cnn_gru_sft": {
        "class": CNNGRUSFTPolicy,
        "ckpt": "best_actor_cnn_gru_sft.pth",
    },
    "fc_lstm_sft": {
        "class": FCLSTMSFTPolicy,
        "ckpt": "best_actor_fc_lstm_sft.pth",
    },
}


_MODEL_CACHE: Dict[Tuple[str, str], Tuple[nn.Module, torch.device]] = {}
_LAST_ACTION = (0.0, 0.0)


def reset_control_state():
    global _LAST_ACTION
    _LAST_ACTION = (0.0, 0.0)


def safe_torch_load(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in [
            "policy_state_dict",
            "actor_state_dict",
            "model_state_dict",
            "state_dict",
        ]:
            if key in checkpoint:
                return checkpoint[key]

    return checkpoint


def load_policy(model_name: str, ckpt_dir: str):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(
            f"unknown model_name={model_name}, "
            f"valid={list(MODEL_CONFIGS.keys())}"
        )

    ckpt_dir = os.path.abspath(ckpt_dir)
    cache_key = (model_name, ckpt_dir)

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = MODEL_CONFIGS[model_name]
    model_class = config["class"]
    ckpt_name = config["ckpt"]
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")

    model = model_class().to(device)

    checkpoint = safe_torch_load(ckpt_path, device)
    state_dict = extract_state_dict(checkpoint)

    model.load_state_dict(state_dict)
    model.eval()

    _MODEL_CACHE[cache_key] = (model, device)

    print(f"[Model] loaded: {model_name}")
    print(f"[Model] ckpt: {ckpt_path}")
    print(f"[Model] device: {device}")

    return model, device


# ============================================================
# LiDAR Preprocess
# ============================================================

def preprocess_lidar(scan: Dict[str, Any]) -> Tuple[np.ndarray, float]:
    ranges = np.asarray(scan.get("ranges", []), dtype=np.float32)

    if ranges.size == 0:
        return np.ones(360, dtype=np.float32), float("inf")

    rmin = float(scan.get("range_min", 0.0))
    rmax = float(scan.get("range_max", 3.5))

    if not math.isfinite(rmax) or rmax <= 0:
        rmax = 3.5

    valid_mask = np.isfinite(ranges) & (ranges >= rmin) & (ranges <= rmax)
    ranges = np.where(valid_mask, ranges, rmax)

    sample_idx = np.linspace(0, len(ranges) - 1, 360, dtype=int)
    lidar_obs = ranges[sample_idx]

    lidar_obs = np.clip(
        lidar_obs / max(rmax, 1e-6),
        0.0,
        1.0,
    ).astype(np.float32)

    angle_min = float(scan.get("angle_min", -math.pi))
    angle_increment = float(scan.get("angle_increment", 0.0))

    front_half_angle_rad = 0.70
    front_valid = []

    for idx, dist in enumerate(ranges):
        if not math.isfinite(float(dist)):
            continue

        angle = angle_min + idx * angle_increment

        if abs(angle) <= front_half_angle_rad:
            front_valid.append(float(dist))

    min_front_dist = min(front_valid) if front_valid else float("inf")

    return lidar_obs, min_front_dist


# ============================================================
# Model Control
# ============================================================

def model_control(
    x: float,
    y: float,
    yaw: float,
    gx: float,
    gy: float,
    scan: Dict[str, Any],
    model_name: str,
    ckpt_dir: str,
    max_v: float = 1.0,
    max_w: float = 1.0,
    min_safe_front_dist: float = 0.30,
    use_safety_layer: bool = True,
) -> Tuple[float, float, Dict[str, Any]]:
    global _LAST_ACTION

    model, device = load_policy(model_name, ckpt_dir)

    lidar_obs, min_front_dist = preprocess_lidar(scan)

    dx = gx - x
    dy = gy - y

    dist_to_goal = math.hypot(dx, dy)
    angle_to_goal = wrap_angle(math.atan2(dy, dx) - yaw)

    ux = math.cos(angle_to_goal)
    uy = math.sin(angle_to_goal)

    prev_v, prev_w = _LAST_ACTION

    prev_v_norm = np.clip(
        prev_v / max(max_v, 1e-6),
        0.0,
        1.0,
    )

    prev_w_norm = np.clip(
        prev_w / max(max_w, 1e-6),
        -1.0,
        1.0,
    )

    state_obs = np.array(
        [
            ux,
            uy,
            prev_v_norm,
            prev_w_norm,
        ],
        dtype=np.float32,
    )

    obs = np.concatenate([lidar_obs, state_obs], axis=0)
    obs_tensor = torch.as_tensor(
        obs,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        mean, _ = model(obs_tensor)
        action = mean.squeeze(0).cpu().numpy()

    raw_v = float(np.clip(action[0] * max_v, 0.0, max_v))
    raw_w = float(np.clip(action[1] * max_w, -max_w, max_w))

    v = raw_v
    w = raw_w

    if use_safety_layer:
        heading_w = float(np.clip(1.8 * angle_to_goal, -max_w, max_w))

        if abs(angle_to_goal) > 0.90:
            v = 0.0
            w = heading_w

        if math.isfinite(min_front_dist):
            scale = np.clip(
                (min_front_dist - min_safe_front_dist) / 0.40,
                0.0,
                1.0,
            )
            v *= scale

        if abs(angle_to_goal) < 0.35 and min_front_dist > 0.45:
            v = max(v, 0.08)

        w = float(np.clip(
            0.60 * raw_w + 0.40 * heading_w,
            -max_w,
            max_w,
        ))

        if math.isfinite(min_front_dist) and min_front_dist < min_safe_front_dist:
            v = 0.0

    v = float(np.clip(v, 0.0, max_v))
    w = float(np.clip(w, -max_w, max_w))

    _LAST_ACTION = (v, w)

    info = {
        "ux": ux,
        "uy": uy,
        "dist_to_goal": dist_to_goal,
        "angle_to_goal": angle_to_goal,
        "min_front_dist": min_front_dist,
        "raw_v": raw_v,
        "raw_w": raw_w,
        "v": v,
        "w": w,
    }

    return v, w, info


# ============================================================
# Track Single Goal
# ============================================================

def track_single_goal(
    goal_xy: Tuple[float, float],
    recorder: DataRecorder,
    goal_name: str = "goal",
    model_name: str = "cnn_lstm_sft",
    ckpt_dir: str = "./Models",
    reach_threshold_m: float = 0.30,
    timeout_sec: float = 120.0,
    control_rate_hz: float = 10.0,
    collision_fail_distance_m: float = 0.30,
    max_v: float = 1.0,
    max_w: float = 1.0,
    use_safety_layer: bool = True,
    scene_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not rclpy.ok():
        rclpy.init()

    reset_control_state()

    scene_data = scene_data or {}
    scene_data.setdefault("collision_happened", 0)
    scene_data.setdefault("model_name", model_name)

    gx, gy = float(goal_xy[0]), float(goal_xy[1])

    node = GoalTracker()

    start_time = time.time()
    dt = 1.0 / max(control_rate_hz, 1e-6)

    collision_grace_sec = 1.0
    collision_confirm_count = 3
    collision_count = 0

    final_pose = None
    final_dist = None
    status = "running"

    try:
        load_policy(model_name, ckpt_dir)

        print(f"[Track] goal_name={goal_name}")
        print(f"[Track] goal=({gx:.3f}, {gy:.3f})")
        print(f"[Track] model={model_name}")

        while rclpy.ok():
            loop_start = time.time()
            elapsed = time.time() - start_time

            if elapsed > timeout_sec:
                status = "timeout"
                print(f"[Track] timeout: {goal_name}")
                break

            rclpy.spin_once(node, timeout_sec=0.01)

            if node.pose is None or node.scan is None:
                time.sleep(0.01)
                continue

            x, y, yaw = node.pose
            final_pose = node.pose

            dx = gx - x
            dy = gy - y

            dist_to_goal = math.hypot(dx, dy)
            final_dist = dist_to_goal

            angle_to_goal = wrap_angle(math.atan2(dy, dx) - yaw)

            ux = math.cos(angle_to_goal)
            uy = math.sin(angle_to_goal)

            _, min_front_dist = preprocess_lidar(node.scan)

            if dist_to_goal <= reach_threshold_m:
                status = "reached"
                node.stop_robot()

                recorder.record(
                    goal_name=goal_name,
                    goal_x=gx,
                    goal_y=gy,
                    ux=ux,
                    uy=uy,
                    pose=node.pose,
                    v=0.0,
                    w=0.0,
                    dist_to_goal=dist_to_goal,
                    angle_to_goal=angle_to_goal,
                    min_front_dist=min_front_dist,
                    status=status,
                    extra_data=scene_data,
                )

                print(f"[Track] reached: {goal_name}, dist={dist_to_goal:.3f}")

                return {
                    "success": True,
                    "status": status,
                    "goal_name": goal_name,
                    "goal_xy": (gx, gy),
                    "elapsed_time": elapsed,
                    "final_pose": final_pose,
                    "final_dist": final_dist,
                    "collision_happened": int(scene_data.get("collision_happened", 0)),
                }

            if (
                elapsed >= collision_grace_sec
                and math.isfinite(min_front_dist)
                and min_front_dist <= collision_fail_distance_m
            ):
                collision_count += 1
            else:
                collision_count = 0

            if collision_count >= collision_confirm_count:
                status = "collision"
                scene_data["collision_happened"] = 1
                node.stop_robot()

                recorder.record(
                    goal_name=goal_name,
                    goal_x=gx,
                    goal_y=gy,
                    ux=ux,
                    uy=uy,
                    pose=node.pose,
                    v=0.0,
                    w=0.0,
                    dist_to_goal=dist_to_goal,
                    angle_to_goal=angle_to_goal,
                    min_front_dist=min_front_dist,
                    status=status,
                    extra_data=scene_data,
                )

                print(
                    f"[Track] collision: {goal_name}, "
                    f"min_front_dist={min_front_dist:.3f}"
                )

                return {
                    "success": False,
                    "status": status,
                    "goal_name": goal_name,
                    "goal_xy": (gx, gy),
                    "elapsed_time": elapsed,
                    "final_pose": final_pose,
                    "final_dist": final_dist,
                    "collision_happened": 1,
                }

            v, w, info = model_control(
                x=x,
                y=y,
                yaw=yaw,
                gx=gx,
                gy=gy,
                scan=node.scan,
                model_name=model_name,
                ckpt_dir=ckpt_dir,
                max_v=max_v,
                max_w=max_w,
                min_safe_front_dist=collision_fail_distance_m,
                use_safety_layer=use_safety_layer,
            )

            recorder.record(
                goal_name=goal_name,
                goal_x=gx,
                goal_y=gy,
                ux=info["ux"],
                uy=info["uy"],
                pose=node.pose,
                v=v,
                w=w,
                dist_to_goal=info["dist_to_goal"],
                angle_to_goal=info["angle_to_goal"],
                min_front_dist=info["min_front_dist"],
                status="running",
                extra_data=scene_data,
            )

            node.send_velocity(v, w)

            used_time = time.time() - loop_start
            sleep_time = max(0.0, dt - used_time)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        status = "keyboard_interrupt"
        print("[Track] keyboard interrupt")

    finally:
        reset_control_state()
        node.stop_robot()
        node.destroy_node()

    return {
        "success": False,
        "status": status,
        "goal_name": goal_name,
        "goal_xy": (gx, gy),
        "elapsed_time": time.time() - start_time,
        "final_pose": final_pose,
        "final_dist": final_dist,
        "collision_happened": int(scene_data.get("collision_happened", 0)),
    }


# ============================================================
# Optional Test
# ============================================================

if __name__ == "__main__":
    print("This file provides DataRecorder and track_single_goal.")
    print("Use it from test_empty_world_5_models.py / test_door_world_5_models.py / test_box_world_5_models.py.")