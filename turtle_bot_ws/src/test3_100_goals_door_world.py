#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import math
from typing import Tuple, Optional, Any, Callable

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity, DeleteEntity, GetModelList
from ament_index_python.packages import get_package_share_directory

from gazebo_goal_point import spawn_goal_point, delete_all_goals


# --- 配置参数 ---
TIMEOUT_PER_GOAL = 120.0  # 每个目标的追踪超时时间（秒）

# 外部菱形墙区域: |x - 3| + |y| <= 4
WORLD_CENTER_X = 3.0
WORLD_CENTER_Y = 0.0
WORLD_L1_RADIUS = 4.0
WORLD_MARGIN = 0.45

DOOR_GAP_WIDTH_MIN = 0.6
DOOR_GAP_WIDTH_MAX = 2.5
DOOR_SEGMENT_START_X = 1.0
DOOR_SEGMENT_START_Y = -2.0
DOOR_SEGMENT_END_X = 5.0
DOOR_SEGMENT_END_Y = 2.0
DOOR_TOTAL_LENGTH = math.hypot(DOOR_SEGMENT_END_X - DOOR_SEGMENT_START_X, DOOR_SEGMENT_END_Y - DOOR_SEGMENT_START_Y)
DOOR_WALL_THICKNESS = 0.2
ROBOT_START_X = 0.0
ROBOT_START_Y = 0.0
ROBOT_BODY_RADIUS = 0.16
ROBOT_SAFETY_MARGIN = 0.10
COLLISION_FAIL_DISTANCE_M = 0.30
ROBOT_RESPAWN_Z = 0.01

DOOR_MODEL_NAME = "resizable_door_world"
ROBOT_ENTITY_NAME = "tb3_trial_bot"
ROBOT_DELETE_KEYWORDS = ["turtlebot3", "waffle", "burger", "tb3_trial_bot"]
PRESERVED_MODEL_NAMES = {"ground_plane", "sun"}
SCENE_EXTRA_FIELDS = [
	"trial_id",
	"door_gap_width",
	"door_wall_length_left",
	"door_wall_length_right",
	"robot_start_x",
	"robot_start_y",
	"robot_start_yaw",
	"world_center_x",
	"world_center_y",
	"world_l1_radius",
	"world_margin",
	"collision_fail_distance_m",
	"collision_happened",
	"robot_clearance_to_wall_m",
]


def get_unique_filename(output_dir: str, base_filename: str) -> str:
	"""始终使用递增后缀命名: name_1.ext, name_2.ext, ..."""
	name, ext = os.path.splitext(base_filename)
	index = 1
	candidate = f"{name}_{index}{ext}"

	while os.path.exists(os.path.join(output_dir, candidate)):
		index += 1
		candidate = f"{name}_{index}{ext}"

	return candidate


def is_in_world_safe_region(x: float, y: float, margin: float = WORLD_MARGIN) -> bool:
	"""判定点是否在菱形外墙内，并与墙保持margin。"""
	return abs(x - WORLD_CENTER_X) + abs(y - WORLD_CENTER_Y) <= (WORLD_L1_RADIUS - margin)


def _door_geometry_from_gap(door_gap_width: float):
	"""根据门洞宽度返回左右墙线段端点和墙长。"""
	gap = max(DOOR_GAP_WIDTH_MIN, min(DOOR_GAP_WIDTH_MAX, door_gap_width))
	wall_len = (DOOR_TOTAL_LENGTH - gap) / 2.0
	if wall_len <= 0.0:
		raise ValueError(f"门洞宽度过大，导致墙长无效: gap={gap:.3f}")

	ux = (DOOR_SEGMENT_END_X - DOOR_SEGMENT_START_X) / DOOR_TOTAL_LENGTH
	uy = (DOOR_SEGMENT_END_Y - DOOR_SEGMENT_START_Y) / DOOR_TOTAL_LENGTH

	left_a = (DOOR_SEGMENT_START_X, DOOR_SEGMENT_START_Y)
	left_b = (DOOR_SEGMENT_START_X + ux * wall_len, DOOR_SEGMENT_START_Y + uy * wall_len)
	right_a = (
		DOOR_SEGMENT_START_X + ux * (wall_len + gap),
		DOOR_SEGMENT_START_Y + uy * (wall_len + gap),
	)
	right_b = (DOOR_SEGMENT_END_X, DOOR_SEGMENT_END_Y)
	return left_a, left_b, right_a, right_b, gap, wall_len


def _project_to_door_frame(x: float, y: float) -> Tuple[float, float]:
	"""投影到门墙局部坐标: s沿门墙方向，n为法向带符号距离。"""
	ux = (DOOR_SEGMENT_END_X - DOOR_SEGMENT_START_X) / DOOR_TOTAL_LENGTH
	uy = (DOOR_SEGMENT_END_Y - DOOR_SEGMENT_START_Y) / DOOR_TOTAL_LENGTH
	vx = -uy
	vy = ux
	dx = x - DOOR_SEGMENT_START_X
	dy = y - DOOR_SEGMENT_START_Y
	s = dx * ux + dy * uy
	n = dx * vx + dy * vy
	return s, n


def _point_to_door_walls_distance(x: float, y: float, door_gap_width: float) -> float:
	"""点到左右门墙线段中心线的最近距离。"""
	left_a, left_b, right_a, right_b, _, _ = _door_geometry_from_gap(door_gap_width)
	d_left = _dist_point_to_segment(x, y, left_a[0], left_a[1], left_b[0], left_b[1])
	d_right = _dist_point_to_segment(x, y, right_a[0], right_a[1], right_b[0], right_b[1])
	return min(d_left, d_right)


def is_inside_door_wall_with_margin(
	x: float,
	y: float,
	door_gap_width: float,
	margin: float = 0.30,
) -> bool:
	"""判定点是否落在门墙附近（按墙厚+margin）。"""
	clearance = _point_to_door_walls_distance(x, y, door_gap_width)
	return clearance <= (DOOR_WALL_THICKNESS / 2.0 + margin)


def _dist_point_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
	"""点到线段的最短距离。"""
	abx = bx - ax
	aby = by - ay
	apx = px - ax
	apy = py - ay
	den = abx * abx + aby * aby
	if den <= 1e-9:
		return math.hypot(px - ax, py - ay)
	t = max(0.0, min(1.0, (apx * abx + apy * aby) / den))
	qx = ax + t * abx
	qy = ay + t * aby
	return math.hypot(px - qx, py - qy)


def _segments_intersect(
	a1: Tuple[float, float],
	a2: Tuple[float, float],
	b1: Tuple[float, float],
	b2: Tuple[float, float],
) -> bool:
	"""判定两线段是否相交（含端点接触）。"""

	def orient(p: Tuple[float, float], q: Tuple[float, float], r: Tuple[float, float]) -> float:
		return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

	def on_segment(p: Tuple[float, float], q: Tuple[float, float], r: Tuple[float, float]) -> bool:
		eps = 1e-9
		return (
			min(p[0], r[0]) - eps <= q[0] <= max(p[0], r[0]) + eps
			and min(p[1], r[1]) - eps <= q[1] <= max(p[1], r[1]) + eps
		)

	o1 = orient(a1, a2, b1)
	o2 = orient(a1, a2, b2)
	o3 = orient(b1, b2, a1)
	o4 = orient(b1, b2, a2)

	if (o1 * o2 < 0.0) and (o3 * o4 < 0.0):
		return True

	if abs(o1) <= 1e-9 and on_segment(a1, b1, a2):
		return True
	if abs(o2) <= 1e-9 and on_segment(a1, b2, a2):
		return True
	if abs(o3) <= 1e-9 and on_segment(b1, a1, b2):
		return True
	if abs(o4) <= 1e-9 and on_segment(b1, a2, b2):
		return True

	return False


def _segment_to_segment_distance(
	a1: Tuple[float, float],
	a2: Tuple[float, float],
	b1: Tuple[float, float],
	b2: Tuple[float, float],
) -> float:
	"""两线段最短距离；相交则返回0。"""
	if _segments_intersect(a1, a2, b1, b2):
		return 0.0

	d1 = _dist_point_to_segment(a1[0], a1[1], b1[0], b1[1], b2[0], b2[1])
	d2 = _dist_point_to_segment(a2[0], a2[1], b1[0], b1[1], b2[0], b2[1])
	d3 = _dist_point_to_segment(b1[0], b1[1], a1[0], a1[1], a2[0], a2[1])
	d4 = _dist_point_to_segment(b2[0], b2[1], a1[0], a1[1], a2[0], a2[1])
	return min(d1, d2, d3, d4)


def _robot_clearance_metrics(door_gap_width: float) -> Tuple[float, float]:
	"""返回机器人到门墙和到外墙最近距离。"""
	clear_to_door = _point_to_door_walls_distance(ROBOT_START_X, ROBOT_START_Y, door_gap_width)

	v1 = (WORLD_CENTER_X - WORLD_L1_RADIUS, WORLD_CENTER_Y)
	v2 = (WORLD_CENTER_X, WORLD_CENTER_Y + WORLD_L1_RADIUS)
	v3 = (WORLD_CENTER_X + WORLD_L1_RADIUS, WORLD_CENTER_Y)
	v4 = (WORLD_CENTER_X, WORLD_CENTER_Y - WORLD_L1_RADIUS)
	segments = [(v1, v2), (v2, v3), (v3, v4), (v4, v1)]
	clear_to_wall = min(
		_dist_point_to_segment(ROBOT_START_X, ROBOT_START_Y, a[0], a[1], b[0], b[1])
		for a, b in segments
	)
	return clear_to_door, clear_to_wall


def _robot_clearance_metrics_at_pose(robot_x: float, robot_y: float, door_gap_width: float) -> Tuple[float, float]:
	"""返回指定机器人位置到门墙和到外墙最近距离。"""
	clear_to_door = _point_to_door_walls_distance(robot_x, robot_y, door_gap_width)

	v1 = (WORLD_CENTER_X - WORLD_L1_RADIUS, WORLD_CENTER_Y)
	v2 = (WORLD_CENTER_X, WORLD_CENTER_Y + WORLD_L1_RADIUS)
	v3 = (WORLD_CENTER_X + WORLD_L1_RADIUS, WORLD_CENTER_Y)
	v4 = (WORLD_CENTER_X, WORLD_CENTER_Y - WORLD_L1_RADIUS)
	segments = [(v1, v2), (v2, v3), (v3, v4), (v4, v1)]
	clear_to_wall = min(
		_dist_point_to_segment(robot_x, robot_y, a[0], a[1], b[0], b[1])
		for a, b in segments
	)
	return clear_to_door, clear_to_wall


def is_scene_safe_for_robot(door_gap_width: float) -> bool:
	"""确保障碍和墙体安全距离不会与机器人本体发生碰撞。"""
	required_clearance = ROBOT_BODY_RADIUS + ROBOT_SAFETY_MARGIN
	clear_to_door, clear_to_wall = _robot_clearance_metrics(door_gap_width)
	return (clear_to_door >= required_clearance) and (clear_to_wall >= required_clearance)


def sample_door_gap_width() -> float:
	"""随机采样门洞宽度。"""
	return random.uniform(DOOR_GAP_WIDTH_MIN, DOOR_GAP_WIDTH_MAX)


def sample_safe_robot_pose(door_gap_width: float, max_attempts: int = 500) -> Tuple[float, float, float]:
	"""采样安全机器人位姿，确保不与门墙和外墙碰撞。"""
	required_clearance = ROBOT_BODY_RADIUS + ROBOT_SAFETY_MARGIN
	inner_margin = WORLD_MARGIN + required_clearance

	for _ in range(max_attempts):
		rx = random.uniform(WORLD_CENTER_X - WORLD_L1_RADIUS + inner_margin, WORLD_CENTER_X - 0.4)
		ry = random.uniform(-2.6, 2.6)
		yaw = random.uniform(-math.pi, math.pi)

		if not is_in_world_safe_region(rx, ry, margin=inner_margin):
			continue
		clear_door, clear_wall = _robot_clearance_metrics_at_pose(rx, ry, door_gap_width)
		if clear_door < required_clearance or clear_wall < required_clearance:
			continue
		s, n = _project_to_door_frame(rx, ry)
		if abs(n) < 0.8 and 0.0 <= s <= DOOR_TOTAL_LENGTH:
			continue
		return rx, ry, yaw

	raise RuntimeError("无法采样到安全机器人位姿，请调整场景或安全参数。")


def is_goal_opposite_and_partially_blocked(
	x: float,
	y: float,
	door_gap_width: float,
	robot_x: float,
	robot_y: float,
) -> bool:
	"""目标在门墙另一侧，且机器人到目标连线必须被门墙阻挡。"""
	_, n_robot = _project_to_door_frame(robot_x, robot_y)
	s_goal, n_goal = _project_to_door_frame(x, y)
	if n_robot * n_goal >= -0.10:
		return False

	left_a, left_b, right_a, right_b, _, _ = _door_geometry_from_gap(door_gap_width)
	path_a = (robot_x, robot_y)
	path_b = (x, y)
	block_threshold = DOOR_WALL_THICKNESS / 2.0 + 0.02
	blocked_by_left = _segment_to_segment_distance(path_a, path_b, left_a, left_b) <= block_threshold
	blocked_by_right = _segment_to_segment_distance(path_a, path_b, right_a, right_b) <= block_threshold
	if not (blocked_by_left or blocked_by_right):
		return False

	# 避免目标落在门洞口正中太近位置（过于简单且可能重叠）
	if abs(n_goal) < 0.45 and (DOOR_TOTAL_LENGTH * 0.5 - door_gap_width * 0.5 - 0.15) <= s_goal <= (DOOR_TOTAL_LENGTH * 0.5 + door_gap_width * 0.5 + 0.15):
		return False

	if is_inside_door_wall_with_margin(x, y, door_gap_width, margin=0.10):
		return False

	return True


def sample_valid_goal(
	door_gap_width: float,
	robot_x: float,
	robot_y: float,
	max_attempts: int = 500,
) -> Tuple[float, float]:
	"""采样位于门墙另一侧且需要穿门附近区域的目标点。"""
	for _ in range(max_attempts):
		x = random.uniform(WORLD_CENTER_X - WORLD_L1_RADIUS + WORLD_MARGIN, WORLD_CENTER_X + WORLD_L1_RADIUS - WORLD_MARGIN)
		y = random.uniform(-3.2, 3.2)

		if not is_in_world_safe_region(x, y):
			continue
		if is_inside_door_wall_with_margin(x, y, door_gap_width):
			continue
		if not is_goal_opposite_and_partially_blocked(x, y, door_gap_width, robot_x, robot_y):
			continue
		if math.hypot(x - robot_x, y - robot_y) < 1.2:
			continue
		if (x * x + y * y) < 0.6 * 0.6:
			continue
		return x, y

	raise RuntimeError("无法采样到有效目标点，请检查边界和障碍物参数。")


def _read_sdf_with_door_gap(sdf_template_path: str, door_gap_width: float, model_name: str) -> str:
	with open(sdf_template_path, "r", encoding="utf-8") as f:
		xml = f.read()

	gap = max(DOOR_GAP_WIDTH_MIN, min(DOOR_GAP_WIDTH_MAX, door_gap_width))
	wall_len = (DOOR_TOTAL_LENGTH - gap) / 2.0
	if wall_len <= 0.0:
		raise ValueError(f"门洞宽度过大，导致墙长无效: gap={gap:.3f}")

	ux = (DOOR_SEGMENT_END_X - DOOR_SEGMENT_START_X) / DOOR_TOTAL_LENGTH
	uy = (DOOR_SEGMENT_END_Y - DOOR_SEGMENT_START_Y) / DOOR_TOTAL_LENGTH
	left_center_s = wall_len / 2.0
	right_center_s = wall_len + gap + wall_len / 2.0
	left_x = DOOR_SEGMENT_START_X + ux * left_center_s
	left_y = DOOR_SEGMENT_START_Y + uy * left_center_s
	right_x = DOOR_SEGMENT_START_X + ux * right_center_s
	right_y = DOOR_SEGMENT_START_Y + uy * right_center_s

	xml = xml.replace("{WALL_POS_X_LEFT}", f"{left_x:.3f}")
	xml = xml.replace("{WALL_POS_Y_LEFT}", f"{left_y:.3f}")
	xml = xml.replace("{WALL_POS_X_RIGHT}", f"{right_x:.3f}")
	xml = xml.replace("{WALL_POS_Y_RIGHT}", f"{right_y:.3f}")
	xml = xml.replace("{WALL_LENGTH_LEFT}", f"{wall_len:.3f}")
	xml = xml.replace("{WALL_LENGTH_RIGHT}", f"{wall_len:.3f}")
	xml = xml.replace("<model name='Resizable_door_writable'>", f"<model name='{model_name}'>")
	return xml


def delete_model_if_exists(model_name: str):
	"""若Gazebo里存在同名模型则删除。"""
	node = Node(f"delete_model_node_{model_name}")
	try:
		get_cli = node.create_client(GetModelList, "/get_model_list")
		if not get_cli.wait_for_service(timeout_sec=5.0):
			raise RuntimeError("/get_model_list service not available")

		req = GetModelList.Request()
		fut = get_cli.call_async(req)
		rclpy.spin_until_future_complete(node, fut, timeout_sec=10.0)
		model_list_res = fut.result()
		if model_list_res is None:
			raise RuntimeError("GetModelList调用失败")

		if model_name not in model_list_res.model_names:
			return

		del_cli = node.create_client(DeleteEntity, "/delete_entity")
		if not del_cli.wait_for_service(timeout_sec=5.0):
			raise RuntimeError("/delete_entity service not available")

		del_req = DeleteEntity.Request()
		del_req.name = model_name
		del_fut = del_cli.call_async(del_req)
		rclpy.spin_until_future_complete(node, del_fut, timeout_sec=10.0)
	finally:
		node.destroy_node()


def _load_turtlebot_sdf_xml() -> str:
	"""加载 turtlebot3 SDF，用于每轮重生机器人。"""
	model_env = os.environ.get("TURTLEBOT3_MODEL", "waffle_pi")
	model_folder = model_env if model_env.startswith("turtlebot3_") else f"turtlebot3_{model_env}"

	search_paths = []
	try:
		share_dir = get_package_share_directory("turtlebot3_gazebo")
		search_paths.append(os.path.join(share_dir, "models", model_folder, "model.sdf"))
	except Exception:
		pass

	for p in os.environ.get("GAZEBO_MODEL_PATH", "").split(":"):
		if p.strip():
			search_paths.append(os.path.join(p.strip(), model_folder, "model.sdf"))

	for candidate in search_paths:
		if os.path.exists(candidate):
			with open(candidate, "r", encoding="utf-8") as f:
				return f.read()

	raise FileNotFoundError(
		f"未找到 {model_folder}/model.sdf。请确认已安装 turtlebot3_gazebo 且 TURTLEBOT3_MODEL 环境变量正确。"
	)


def delete_existing_robot_models():
	"""删除Gazebo内已有机器人，避免多机器人话题冲突。"""
	node = Node("delete_existing_robot_models_node")
	try:
		get_cli = node.create_client(GetModelList, "/get_model_list")
		if not get_cli.wait_for_service(timeout_sec=5.0):
			raise RuntimeError("/get_model_list service not available")

		fut = get_cli.call_async(GetModelList.Request())
		rclpy.spin_until_future_complete(node, fut, timeout_sec=10.0)
		res = fut.result()
		if res is None:
			raise RuntimeError("GetModelList调用失败")

		targets = []
		for name in res.model_names:
			lname = name.lower()
			if any(k in lname for k in ROBOT_DELETE_KEYWORDS):
				targets.append(name)

		if not targets:
			return

		del_cli = node.create_client(DeleteEntity, "/delete_entity")
		if not del_cli.wait_for_service(timeout_sec=5.0):
			raise RuntimeError("/delete_entity service not available")

		for name in targets:
			req = DeleteEntity.Request()
			req.name = name
			del_fut = del_cli.call_async(req)
			rclpy.spin_until_future_complete(node, del_fut, timeout_sec=10.0)
	finally:
		node.destroy_node()


def clear_all_world_models(preserved_model_names: Optional[set] = None):
	"""清空world里除保留名单外的所有模型，避免场景切换残留重叠。"""
	preserved = preserved_model_names or PRESERVED_MODEL_NAMES
	node = Node("clear_all_world_models_node")
	try:
		get_cli = node.create_client(GetModelList, "/get_model_list")
		if not get_cli.wait_for_service(timeout_sec=5.0):
			raise RuntimeError("/get_model_list service not available")

		fut = get_cli.call_async(GetModelList.Request())
		rclpy.spin_until_future_complete(node, fut, timeout_sec=10.0)
		res = fut.result()
		if res is None:
			raise RuntimeError("GetModelList调用失败")

		targets = [name for name in res.model_names if name not in preserved]
		if not targets:
			return

		del_cli = node.create_client(DeleteEntity, "/delete_entity")
		if not del_cli.wait_for_service(timeout_sec=5.0):
			raise RuntimeError("/delete_entity service not available")

		for name in targets:
			req = DeleteEntity.Request()
			req.name = name
			del_fut = del_cli.call_async(req)
			rclpy.spin_until_future_complete(node, del_fut, timeout_sec=10.0)
	finally:
		node.destroy_node()


def spawn_robot_entity(robot_x: float, robot_y: float, robot_yaw: float):
	"""按给定位姿重生机器人。"""
	node = Node("spawn_robot_entity_node")
	try:
		spawn_cli = node.create_client(SpawnEntity, "/spawn_entity")
		if not spawn_cli.wait_for_service(timeout_sec=10.0):
			raise RuntimeError("/spawn_entity service not available. Is Gazebo running?")

		sdf_xml = _load_turtlebot_sdf_xml()
		req = SpawnEntity.Request()
		req.name = ROBOT_ENTITY_NAME
		req.xml = sdf_xml
		req.initial_pose.position.x = float(robot_x)
		req.initial_pose.position.y = float(robot_y)
		req.initial_pose.position.z = float(ROBOT_RESPAWN_Z)
		req.initial_pose.orientation.z = math.sin(robot_yaw / 2.0)
		req.initial_pose.orientation.w = math.cos(robot_yaw / 2.0)

		fut = spawn_cli.call_async(req)
		rclpy.spin_until_future_complete(node, fut, timeout_sec=15.0)
		if fut.result() is None:
			raise RuntimeError("机器人SpawnEntity调用失败")
	finally:
		node.destroy_node()


def spawn_door_world_obstacle(sdf_template_path: str, door_gap_width: float, model_name: str = DOOR_MODEL_NAME):
	"""用模板SDF按门洞宽度生成并加载DOOR世界障碍模型。"""
	node = Node(f"spawn_door_node_{model_name}")
	try:
		delete_model_if_exists(model_name)

		spawn_cli = node.create_client(SpawnEntity, "/spawn_entity")
		if not spawn_cli.wait_for_service(timeout_sec=10.0):
			raise RuntimeError("/spawn_entity service not available. Is Gazebo running?")

		req = SpawnEntity.Request()
		req.name = model_name
		req.xml = _read_sdf_with_door_gap(sdf_template_path, door_gap_width, model_name)
		req.initial_pose.orientation.w = 1.0

		fut = spawn_cli.call_async(req)
		rclpy.spin_until_future_complete(node, fut, timeout_sec=10.0)
		if fut.result() is None:
			raise RuntimeError("SpawnEntity调用失败")
	finally:
		node.destroy_node()


def StartTest_DoorWorld(ModelName: str, NUM_TRIALS: int = 100):
	if ModelName == "Model_1_PPO_Ckpt_Step_10000":
		from chase_goal_record_data_Model1PpoCkptStep10000 import (
			track_single_goal as _track_single_goal,
			DataRecorder as _DataRecorder,
		)
		base_file_name = "Output_Model_1_PPO_Ckpt_Door_World_Test_100.csv"
	elif ModelName == "Model_2_PPO_Ckpt_Step_10000":
		from chase_goal_record_data_PpoCkptStep10000 import (
			track_single_goal as _track_single_goal,
			DataRecorder as _DataRecorder,
		)
		base_file_name = "Output_Model_2_PPO_Ckpt_Door_World_Test_100.csv"
	else:
		# default to Model 1 if unknown model name is provided
		from chase_goal_record_data_Model1PpoCkptStep10000 import (
			track_single_goal as _track_single_goal,
			DataRecorder as _DataRecorder,
		)
		base_file_name = "Output_Model_1_PPO_Ckpt_Door_World_Test_100.csv"

	# 统一引用，避免分支导入后产生 Union 类型冲突。
	track_single_goal_fn: Callable[..., bool] = _track_single_goal
	DataRecorderCls: type[Any] = _DataRecorder

	rclpy.init()

	try:
		clear_all_world_models()
		delete_all_goals()

		output_dir = os.path.join(os.path.dirname(__file__), "Models", "Outputs")
		os.makedirs(output_dir, exist_ok=True)
		file_name = get_unique_filename(output_dir, base_file_name)
		recorder = DataRecorderCls(
			filename=os.path.join(output_dir, file_name),
			extra_fields=SCENE_EXTRA_FIELDS,
		)

		sdf_template_path = os.path.join(os.path.dirname(__file__), "Worlds", "Resizable_door_writable.sdf")
		if not os.path.exists(sdf_template_path):
			raise FileNotFoundError(f"未找到SDF模板文件: {sdf_template_path}")

		print(f"开始进行 {NUM_TRIALS} 轮 door_world 随机目标追踪实验...")

		for i in range(1, NUM_TRIALS + 1):
			clear_all_world_models()
			delete_all_goals()

			door_gap_width = sample_door_gap_width()
			wall_len = (DOOR_TOTAL_LENGTH - door_gap_width) / 2.0
			robot_x, robot_y, robot_yaw = sample_safe_robot_pose(door_gap_width)
			_, robot_clearance_to_wall = _robot_clearance_metrics_at_pose(
				robot_x,
				robot_y,
				door_gap_width,
			)

			spawn_door_world_obstacle(
				sdf_template_path=sdf_template_path,
				door_gap_width=door_gap_width,
				model_name=DOOR_MODEL_NAME,
			)
			spawn_robot_entity(robot_x, robot_y, robot_yaw)

			random_x, random_y = sample_valid_goal(door_gap_width, robot_x, robot_y)
			goal_name = f"door_goal_{i}"

			print(
				f"\n--- 第{i}/{NUM_TRIALS}轮 ---",
				f"\n{ModelName}",
				"\nDOOR WORLD",
				# f"\nrobot ({robot_x:.2f}, {robot_y:.2f}, yaw={robot_yaw:.2f})",
				f"\n{goal_name} ({random_x:.2f}, {random_y:.2f})",
				f"\ndoor_gap={door_gap_width:.2f}, wall_len={wall_len:.2f}"
			)

			spawn_goal_point(random_x, random_y, 0.2, name=goal_name)

			reached = track_single_goal_fn(
				goal_xy=(random_x, random_y),
				recorder=recorder,
				goal_name=goal_name,
				timeout_sec=TIMEOUT_PER_GOAL,
				reach_threshold_m=0.3,
				collision_fail_distance_m=COLLISION_FAIL_DISTANCE_M,
				scene_data={
					"trial_id": i,
					"door_gap_width": round(door_gap_width, 4),
					"door_wall_length_left": round(wall_len, 4),
					"door_wall_length_right": round(wall_len, 4),
					"robot_start_x": round(robot_x, 4),
					"robot_start_y": round(robot_y, 4),
					"robot_start_yaw": round(robot_yaw, 4),
					"world_center_x": WORLD_CENTER_X,
					"world_center_y": WORLD_CENTER_Y,
					"world_l1_radius": WORLD_L1_RADIUS,
					"world_margin": WORLD_MARGIN,
					"collision_fail_distance_m": COLLISION_FAIL_DISTANCE_M,
					"collision_happened": 0,
					"robot_clearance_to_wall_m": round(robot_clearance_to_wall, 4),
				},
			)

			if reached:
				print(f"成功: {goal_name} 已到达")
			else:
				print(f"失败: {goal_name} 未到达")

			delete_all_goals()

		print(f"\n所有实验已完成。数据记录在 '{os.path.join(output_dir, file_name)}' 中。\n")

	finally:
		try:
			clear_all_world_models()
			delete_all_goals()
		except Exception as e:
			print(f"收尾清理时出现异常: {e}\n")
		rclpy.shutdown()


if __name__ == "__main__":
	StartTest_DoorWorld("Model_1_PPO_Ckpt_Step_10000")
	StartTest_DoorWorld("Model_2_PPO_Ckpt_Step_10000")
