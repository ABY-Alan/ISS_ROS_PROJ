#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Tuple, Optional, Dict, Any
import math
import time

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import random

def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    # yaw from quaternion (Z-axis rotation), no external deps
    # yaw = atan2(2(wz + xy), 1 - 2(y^2 + z^2))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class _GoalTracker(Node):
    def __init__(self):
        super().__init__("goal_tracker_min")

        # /odom
        self._odom_sub = self.create_subscription(Odometry, "/odom", self._on_odom, 10)
        self._last_pose: Optional[Tuple[float, float, float]] = None  # (x, y, yaw)

        # /scan
        self._scan_sub = self.create_subscription(LaserScan, "/scan", self._on_scan, 10)
        self._last_scan: Optional[Dict[str, Any]] = None  # minimal scan dict

        # /cmd_vel
        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

    def _on_odom(self, msg: Odometry):
        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        yaw = quat_to_yaw(float(q.x), float(q.y), float(q.z), float(q.w))
        self._last_pose = (x, y, yaw)

    def _on_scan(self, msg: LaserScan):
        # 只存最关键的输入，避免直接存 ROS message（也可以存msg本身）
        self._last_scan = {
            "ranges": list(msg.ranges),
            "angle_min": float(msg.angle_min),
            "angle_increment": float(msg.angle_increment),
            "range_min": float(msg.range_min),
            "range_max": float(msg.range_max),
        }

    def get_pose(self) -> Optional[Tuple[float, float, float]]:
        return self._last_pose

    def get_scan(self) -> Optional[Dict[str, Any]]:
        return self._last_scan

    def publish_cmd(self, v: float, w: float):
        t = Twist()
        t.linear.x = float(v)
        t.angular.z = float(w)
        self._cmd_pub.publish(t)

    def stop_robot(self):
        self.publish_cmd(0.0, 0.0)


def control_policy_no_model(
    x: float,
    y: float,
    yaw: float,
    gx: float,
    gy: float,
    scan: Dict[str, Any],
) -> Tuple[float, float]:

    # ===== 1) 车 -> goal 向量与角度误差 =====
    dx = gx - x
    dy = gy - y
    goal_yaw = math.atan2(dy, dx)

    angle_error = goal_yaw - yaw
    angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))  # wrap to [-pi, pi]

    # 角速度（P 控制）
    k_w = 2.0
    max_w = 1.5
    w = max(-max_w, min(max_w, k_w * angle_error))

    # ===== 2) 统计 scan 平均障碍物距离 =====
    ranges = scan["ranges"]
    rmin = scan["range_min"]
    rmax = scan["range_max"]

    valid = [r for r in ranges if (r is not None) and math.isfinite(r) and (rmin <= r <= rmax)]
    # 如果没有有效数据，给一个很大的平均值（当作“很安全”）
    avg_dist = (sum(valid) / len(valid)) if valid else rmax

    # ===== 3) 用平均距离设置 v =====
    # 你可以按需调这几个阈值
    stop_dist = 0.35   # 平均距离小于这个就停
    full_dist = 1.20   # 平均距离大于这个就全速

    v_max = 0.22       # 最大前进速度
    v_min = 0.00       # 最小前进速度（停）

    # 如果角度误差大，先转向，不前进（可选但建议）
    angle_threshold = 0.35  # ~20°
    if abs(angle_error) > angle_threshold:
        v = 0.0
        return v, w

    if avg_dist <= stop_dist:
        v = 0.0
    elif avg_dist >= full_dist:
        v = v_max
    else:
        # 线性插值：avg_dist 从 stop_dist->full_dist 映射到 v_min->v_max
        ratio = (avg_dist - stop_dist) / (full_dist - stop_dist)
        v = v_min + ratio * (v_max - v_min)

    return float(v), float(w)

def control_policy_no_model_with_noise(
    x: float,
    y: float,
    yaw: float,
    gx: float,
    gy: float,
    scan: Dict[str, Any],
) -> Tuple[float, float]:

    # ===== 1) 车 -> goal 向量与角度误差 =====
    dx = gx - x
    dy = gy - y
    goal_yaw = math.atan2(dy, dx)

    angle_error = goal_yaw - yaw
    angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))

    # 角速度 P 控制
    k_w = 2.0
    max_w = 1.5
    w = max(-max_w, min(max_w, k_w * angle_error))

    # ===== 2) 统计 scan 平均障碍物距离 =====
    ranges = scan["ranges"]
    rmin = scan["range_min"]
    rmax = scan["range_max"]

    valid = [r for r in ranges if (r is not None) and math.isfinite(r) and (rmin <= r <= rmax)]
    avg_dist = (sum(valid) / len(valid)) if valid else rmax

    # ===== 3) 根据平均距离设置 v =====
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

    # ===== 4) 加入随机抖动（执行器噪声模拟） =====

    # 高斯噪声（更真实）
    v_noise_std = 0.5
    w_noise_std = 0.3

    v += random.gauss(0.0, v_noise_std)
    w += random.gauss(0.0, w_noise_std)

    # 限幅，防止失控
    v = max(0.0, min(v_max, v))
    w = max(-max_w, min(max_w, w))

    return float(v), float(w)

def track_single_goal(
    goal_xy: Tuple[float, float],
    reach_threshold_m: float = 0.25,
    timeout_sec: float = 120.0,
    control_rate_hz: float = 10.0,
) -> bool:
    rclpy.init(args=None)
    node = _GoalTracker()

    gx, gy = float(goal_xy[0]), float(goal_xy[1])
    dt = 1.0 / float(control_rate_hz)
    t0 = time.time()

    try:
        while rclpy.ok():
            # 让回调跑起来，更新 /odom 和 /scan
            rclpy.spin_once(node, timeout_sec=0.1)

            pose = node.get_pose()
            scan = node.get_scan()
            if pose is None or scan is None:
                continue

            x, y, yaw = pose
            dist = math.hypot(gx - x, gy - y)

            # 停止条件：到达 goal
            if dist <= reach_threshold_m:
                node.stop_robot()
                return True

            # 停止条件：超时
            if (time.time() - t0) >= float(timeout_sec):
                node.stop_robot()
                return False

            # 控制逻辑：现在 input 已经齐全
            v, w = control_policy_no_model_with_noise(x, y, yaw, gx, gy, scan)
            node.publish_cmd(v, w)

            time.sleep(dt)

    finally:
        try:
            node.stop_robot()
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()