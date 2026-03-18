#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import math
import time
from typing import Tuple, Optional, Any, Callable

import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import SpawnEntity, DeleteEntity, GetModelList
from ament_index_python.packages import get_package_share_directory

from gazebo_goal_point import spawn_goal_point, delete_all_goals


# --- 配置参数 ---
TIMEOUT_PER_GOAL = 300.0

# 外部菱形墙区域: |x - 3| + |y| <= 4
WORLD_CENTER_X = 3.0
WORLD_CENTER_Y = 0.0
WORLD_L1_RADIUS = 4.0
WORLD_MARGIN = 0.45

# 可变BOX尺寸范围（与 SDF 注释一致）
BOX_SIZE_X_MIN = 0.5
BOX_SIZE_X_MAX = 2.0
BOX_SIZE_Y_MIN = 1.0
BOX_SIZE_Y_MAX = 2.5
BOX_YAW_RAD = 0.785398  # 45度
ROBOT_START_X = 0.0
ROBOT_START_Y = 0.0
ROBOT_BODY_RADIUS = 0.16
ROBOT_SAFETY_MARGIN = 0.10
COLLISION_FAIL_DISTANCE_M = 0.30
ROBOT_RESPAWN_Z = 0.01

BOX_MODEL_NAME = "resizable_box_world"
ROBOT_ENTITY_NAME = "tb3_trial_bot"
ROBOT_DELETE_KEYWORDS = ["turtlebot3", "waffle", "burger", "tb3_trial_bot"]
PRESERVED_MODEL_NAMES = {"ground_plane", "sun"}
SCENE_EXTRA_FIELDS = [
	"trial_id",
	"box_size_x",
	"box_size_y",
	"box_yaw_rad",
	"robot_start_x",
	"robot_start_y",
	"robot_start_yaw",
	"box_center_x",
	"box_center_y",
	"world_center_x",
	"world_center_y",
	"world_l1_radius",
	"world_margin",
	"collision_fail_distance_m",
	"collision_happened",
	"robot_clearance_to_box_m",
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


def _format_seconds(seconds: float) -> str:
	total_seconds = max(0, int(seconds))
	hours, remainder = divmod(total_seconds, 3600)
	minutes, secs = divmod(remainder, 60)
	return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _rotate_world_to_local(dx: float, dy: float, yaw: float) -> Tuple[float, float]:
	"""把世界坐标偏移量旋转到BOX局部坐标系。"""
	c = math.cos(yaw)
	s = math.sin(yaw)
	# 旋转 -yaw
	lx = c * dx + s * dy
	ly = -s * dx + c * dy
	return lx, ly


def is_in_world_safe_region(x: float, y: float, margin: float = WORLD_MARGIN) -> bool:
	"""判定点是否在菱形外墙内，并与墙保持margin。"""
	return abs(x - WORLD_CENTER_X) + abs(y - WORLD_CENTER_Y) <= (WORLD_L1_RADIUS - margin)


def is_inside_box_with_margin(
	x: float,
	y: float,
	box_size_x: float,
	box_size_y: float,
	margin: float = 0.30,
) -> bool:
	"""判定点是否落在旋转BOX内（加安全margin）。"""
	dx = x - WORLD_CENTER_X
	dy = y - WORLD_CENTER_Y
	lx, ly = _rotate_world_to_local(dx, dy, BOX_YAW_RAD)
	return abs(lx) <= (box_size_x / 2.0 + margin) and abs(ly) <= (box_size_y / 2.0 + margin)


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


def _point_to_rotated_box_distance(
	px: float,
	py: float,
	box_cx: float,
	box_cy: float,
	box_sx: float,
	box_sy: float,
	box_yaw: float,
) -> float:
	"""点到旋转矩形边界的最短欧氏距离（点在内部时返回0）。"""
	dx = px - box_cx
	dy = py - box_cy
	lx, ly = _rotate_world_to_local(dx, dy, box_yaw)
	qx = max(abs(lx) - box_sx / 2.0, 0.0)
	qy = max(abs(ly) - box_sy / 2.0, 0.0)
	return math.hypot(qx, qy)


def _robot_clearance_metrics(box_size_x: float, box_size_y: float) -> Tuple[float, float]:
	"""返回机器人到box和到外墙最近距离。"""
	clear_to_box = _point_to_rotated_box_distance(
		ROBOT_START_X,
		ROBOT_START_Y,
		WORLD_CENTER_X,
		WORLD_CENTER_Y,
		box_size_x,
		box_size_y,
		BOX_YAW_RAD,
	)

	v1 = (WORLD_CENTER_X - WORLD_L1_RADIUS, WORLD_CENTER_Y)
	v2 = (WORLD_CENTER_X, WORLD_CENTER_Y + WORLD_L1_RADIUS)
	v3 = (WORLD_CENTER_X + WORLD_L1_RADIUS, WORLD_CENTER_Y)
	v4 = (WORLD_CENTER_X, WORLD_CENTER_Y - WORLD_L1_RADIUS)
	segments = [(v1, v2), (v2, v3), (v3, v4), (v4, v1)]
	clear_to_wall = min(
		_dist_point_to_segment(ROBOT_START_X, ROBOT_START_Y, a[0], a[1], b[0], b[1])
		for a, b in segments
	)
	return clear_to_box, clear_to_wall


def _robot_clearance_metrics_at_pose(robot_x: float, robot_y: float, box_size_x: float, box_size_y: float) -> Tuple[float, float]:
	"""返回指定机器人位置到box和到外墙最近距离。"""
	clear_to_box = _point_to_rotated_box_distance(
		robot_x,
		robot_y,
		WORLD_CENTER_X,
		WORLD_CENTER_Y,
		box_size_x,
		box_size_y,
		BOX_YAW_RAD,
	)

	v1 = (WORLD_CENTER_X - WORLD_L1_RADIUS, WORLD_CENTER_Y)
	v2 = (WORLD_CENTER_X, WORLD_CENTER_Y + WORLD_L1_RADIUS)
	v3 = (WORLD_CENTER_X + WORLD_L1_RADIUS, WORLD_CENTER_Y)
	v4 = (WORLD_CENTER_X, WORLD_CENTER_Y - WORLD_L1_RADIUS)
	segments = [(v1, v2), (v2, v3), (v3, v4), (v4, v1)]
	clear_to_wall = min(
		_dist_point_to_segment(robot_x, robot_y, a[0], a[1], b[0], b[1])
		for a, b in segments
	)
	return clear_to_box, clear_to_wall


def is_scene_safe_for_robot(box_size_x: float, box_size_y: float) -> bool:
	"""确保障碍和墙体安全距离不会与机器人本体发生碰撞。"""
	required_clearance = ROBOT_BODY_RADIUS + ROBOT_SAFETY_MARGIN
	clear_to_box, clear_to_wall = _robot_clearance_metrics(box_size_x, box_size_y)
	return (clear_to_box >= required_clearance) and (clear_to_wall >= required_clearance)


def sample_safe_box_size(max_attempts: int = 200) -> Tuple[float, float]:
	"""采样满足机器人安全间距约束的box尺寸。"""
	for _ in range(max_attempts):
		sx = random.uniform(BOX_SIZE_X_MIN, BOX_SIZE_X_MAX)
		sy = random.uniform(BOX_SIZE_Y_MIN, BOX_SIZE_Y_MAX)
		if is_scene_safe_for_robot(sx, sy):
			return sx, sy
	raise RuntimeError("无法采样到满足机器人安全距离约束的BOX尺寸，请调整参数范围。")


def sample_safe_robot_pose(box_size_x: float, box_size_y: float, max_attempts: int = 500) -> Tuple[float, float, float]:
	"""采样安全机器人位姿，确保不与box和外墙碰撞。"""
	required_clearance = ROBOT_BODY_RADIUS + ROBOT_SAFETY_MARGIN
	inner_margin = WORLD_MARGIN + required_clearance

	for _ in range(max_attempts):
		rx = random.uniform(WORLD_CENTER_X - WORLD_L1_RADIUS + inner_margin, WORLD_CENTER_X + 0.4)
		ry = random.uniform(-2.6, 2.6)
		yaw = random.uniform(-math.pi, math.pi)

		if not is_in_world_safe_region(rx, ry, margin=inner_margin):
			continue
		clear_box, clear_wall = _robot_clearance_metrics_at_pose(rx, ry, box_size_x, box_size_y)
		if clear_box < required_clearance or clear_wall < required_clearance:
			continue
		if math.hypot(rx - WORLD_CENTER_X, ry - WORLD_CENTER_Y) < 0.8:
			continue
		return rx, ry, yaw

	raise RuntimeError("无法采样到安全机器人位姿，请调整场景或安全参数。")


def is_goal_opposite_and_partially_blocked(
	x: float,
	y: float,
	box_size_x: float,
	box_size_y: float,
	robot_x: float,
	robot_y: float,
) -> bool:
	"""目标在box相对机器人对面，且机器人到目标连线会被box部分遮挡。"""
	rvx = robot_x - WORLD_CENTER_X
	rvy = robot_y - WORLD_CENTER_Y
	gvx = x - WORLD_CENTER_X
	gvy = y - WORLD_CENTER_Y
	# 对面：目标向量与机器人向量夹角大于90度
	if (gvx * rvx + gvy * rvy) >= -0.25:
		return False

	# 略微挡住：机器人到目标连线需靠近box中心
	path_dist = _dist_point_to_segment(
		WORLD_CENTER_X,
		WORLD_CENTER_Y,
		robot_x,
		robot_y,
		x,
		y,
	)
	box_half_diag = 0.5 * math.hypot(box_size_x, box_size_y)
	max_allow_dist = box_half_diag + 0.30
	if path_dist > max_allow_dist:
		return False

	# 避免目标过于贴近box正后方边缘，增加可达性
	dx = x - WORLD_CENTER_X
	dy = y - WORLD_CENTER_Y
	lx, ly = _rotate_world_to_local(dx, dy, BOX_YAW_RAD)
	if abs(lx) <= (box_size_x / 2.0 + 0.15) and abs(ly) <= (box_size_y / 2.0 + 0.60):
		return False

	return True


def sample_valid_goal(
	box_size_x: float,
	box_size_y: float,
	robot_x: float,
	robot_y: float,
	max_attempts: int = 500,
) -> Tuple[float, float]:
	"""采样位于box对面且路径被box略遮挡的有效目标点。"""
	for _ in range(max_attempts):
		x = random.uniform(WORLD_CENTER_X - WORLD_L1_RADIUS + WORLD_MARGIN, WORLD_CENTER_X + WORLD_L1_RADIUS - WORLD_MARGIN)
		y = random.uniform(-3.2, 3.2)

		if not is_in_world_safe_region(x, y):
			continue
		if is_inside_box_with_margin(x, y, box_size_x, box_size_y):
			continue
		if not is_goal_opposite_and_partially_blocked(x, y, box_size_x, box_size_y, robot_x, robot_y):
			continue
		if math.hypot(x - robot_x, y - robot_y) < 1.2:
			continue
		if (x * x + y * y) < 0.6 * 0.6:
			continue
		return x, y

	raise RuntimeError("无法采样到有效目标点，请检查边界和障碍物参数。")


def _read_sdf_with_size(sdf_template_path: str, size_x: float, size_y: float, model_name: str) -> str:
	with open(sdf_template_path, "r", encoding="utf-8") as f:
		xml = f.read()

	xml = xml.replace("{SIZE_X}", f"{size_x:.3f}")
	xml = xml.replace("{SIZE_Y}", f"{size_y:.3f}")
	xml = xml.replace("<model name='Resizable_box'>", f"<model name='{model_name}'>")
	xml = xml.replace("<model name='Resizable_box_writable'>", f"<model name='{model_name}'>")
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


def spawn_box_world_obstacle(sdf_template_path: str, size_x: float, size_y: float, model_name: str = BOX_MODEL_NAME):
	"""用模板SDF按指定尺寸生成并加载BOX世界障碍模型。"""
	node = Node(f"spawn_box_node_{model_name}")
	try:
		delete_model_if_exists(model_name)

		spawn_cli = node.create_client(SpawnEntity, "/spawn_entity")
		if not spawn_cli.wait_for_service(timeout_sec=10.0):
			raise RuntimeError("/spawn_entity service not available. Is Gazebo running?")

		req = SpawnEntity.Request()
		req.name = model_name
		req.xml = _read_sdf_with_size(sdf_template_path, size_x, size_y, model_name)
		req.initial_pose.orientation.w = 1.0

		fut = spawn_cli.call_async(req)
		rclpy.spin_until_future_complete(node, fut, timeout_sec=10.0)
		if fut.result() is None:
			raise RuntimeError("SpawnEntity调用失败")
	finally:
		node.destroy_node()


def StartTest_BoxWorld(ModelName: str, NUM_TRIALS: int = 200):
	if ModelName == "Model_1_PPO_Ckpt_Step_10000":
		from chase_goal_record_data_Model1PpoCkptStep10000 import (
			track_single_goal as _track_single_goal,
			DataRecorder as _DataRecorder,
		)
		base_file_name = "Output_Model_1_PPO_Ckpt_Box_World_Test_100.csv"
	elif ModelName == "Model_2_PPO_Ckpt_Step_10000":
		from chase_goal_record_data_PpoCkptStep10000 import (
			track_single_goal as _track_single_goal,
			DataRecorder as _DataRecorder,
		)
		base_file_name = "Output_Model_2_PPO_Ckpt_Box_World_Test_100.csv"
	else:
		# default to Model 1 if unknown model name is provided
		from chase_goal_record_data_Model1PpoCkptStep10000 import (
			track_single_goal as _track_single_goal,
			DataRecorder as _DataRecorder,
		)
		base_file_name = "Output_Model_1_PPO_Ckpt_Box_World_Test_100.csv"

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

		sdf_template_path = os.path.join(os.path.dirname(__file__), "Worlds", "Resizable_box_writable.sdf")
		if not os.path.exists(sdf_template_path):
			raise FileNotFoundError(f"未找到SDF模板文件: {sdf_template_path}")

		print(f"开始进行 {NUM_TRIALS} 轮 box_world 随机目标追踪实验...")
		total_start_time = time.perf_counter()

		for i in range(1, NUM_TRIALS + 1):
			round_start_time = time.perf_counter()
			clear_all_world_models()
			delete_all_goals()

			box_size_x, box_size_y = sample_safe_box_size()
			robot_x, robot_y, robot_yaw = sample_safe_robot_pose(box_size_x, box_size_y)
			robot_clearance_to_box, robot_clearance_to_wall = _robot_clearance_metrics_at_pose(
				robot_x,
				robot_y,
				box_size_x,
				box_size_y,
			)

			spawn_box_world_obstacle(
				sdf_template_path=sdf_template_path,
				size_x=box_size_x,
				size_y=box_size_y,
				model_name=BOX_MODEL_NAME,
			)
			spawn_robot_entity(robot_x, robot_y, robot_yaw)

			random_x, random_y = sample_valid_goal(box_size_x, box_size_y, robot_x, robot_y)
			goal_name = f"box_goal_{i}"

			print(
				f"\n--- 第{i}/{NUM_TRIALS}轮 ---",
				f"\n{ModelName}",
				"\nBOX WORLD",
				f"\nrobot ({robot_x:.2f}, {robot_y:.2f}, yaw={robot_yaw:.2f})",
				f"\n{goal_name} ({random_x:.2f}, {random_y:.2f})",
				f"\nbox_size=({box_size_x:.2f}, {box_size_y:.2f})"
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
					"box_size_x": round(box_size_x, 4),
					"box_size_y": round(box_size_y, 4),
					"box_yaw_rad": BOX_YAW_RAD,
					"robot_start_x": round(robot_x, 4),
					"robot_start_y": round(robot_y, 4),
					"robot_start_yaw": round(robot_yaw, 4),
					"box_center_x": WORLD_CENTER_X,
					"box_center_y": WORLD_CENTER_Y,
					"world_center_x": WORLD_CENTER_X,
					"world_center_y": WORLD_CENTER_Y,
					"world_l1_radius": WORLD_L1_RADIUS,
					"world_margin": WORLD_MARGIN,
					"collision_fail_distance_m": COLLISION_FAIL_DISTANCE_M,
					"collision_happened": 0,
					"robot_clearance_to_box_m": round(robot_clearance_to_box, 4),
					"robot_clearance_to_wall_m": round(robot_clearance_to_wall, 4),
				},
			)

			if reached:
				print(f"成功: {goal_name} 已到达")
			else:
				print(f"失败: {goal_name} 未到达")

			delete_all_goals()

			round_elapsed = time.perf_counter() - round_start_time
			total_elapsed = time.perf_counter() - total_start_time

			print(
				f"耗时统计: "
				f"本轮={_format_seconds(round_elapsed)} | "
				f"累计={_format_seconds(total_elapsed)} | "
			)

		print(f"\n所有实验已完成。数据记录在 '{os.path.join(output_dir, file_name)}' 中。\n")

	finally:
		try:
			clear_all_world_models()
			delete_all_goals()
		except Exception as e:
			print(f"收尾清理时出现异常: {e}\n")
		rclpy.shutdown()


if __name__ == "__main__":
	StartTest_BoxWorld("Model_1_PPO_Ckpt_Step_10000")
	StartTest_BoxWorld("Model_2_PPO_Ckpt_Step_10000")
