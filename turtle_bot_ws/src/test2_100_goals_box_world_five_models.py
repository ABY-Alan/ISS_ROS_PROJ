#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time

import rclpy

import test2_100_goals_box_world as box_world
from chase_goal_record_data_PpoMixedFiveModels import (
    DataRecorder,
    get_output_tag,
    get_supported_model_names,
    track_single_goal,
)
from gazebo_goal_point import delete_all_goals, spawn_goal_point


TIMEOUT_PER_GOAL = box_world.TIMEOUT_PER_GOAL


def StartTest_BoxWorld_FiveModels(model_name: str, NUM_TRIALS: int = 200):
    if model_name not in get_supported_model_names():
        supported = ", ".join(get_supported_model_names())
        raise ValueError(f"不支持的模型名: {model_name}。可选值: {supported}")

    base_file_name = f"Output_{get_output_tag(model_name)}_Box_World_Test.csv"

    rclpy.init()

    try:
        box_world.clear_all_world_models()
        delete_all_goals()

        output_dir = os.path.join(os.path.dirname(__file__), "Models", "Outputs")
        os.makedirs(output_dir, exist_ok=True)
        file_name = box_world.get_unique_filename(output_dir, base_file_name)
        recorder = DataRecorder(
            filename=os.path.join(output_dir, file_name),
            extra_fields=box_world.SCENE_EXTRA_FIELDS,
        )

        sdf_template_path = os.path.join(os.path.dirname(__file__), "Worlds", "Resizable_box_writable.sdf")
        if not os.path.exists(sdf_template_path):
            raise FileNotFoundError(f"未找到SDF模板文件: {sdf_template_path}")

        print(f"开始进行 {NUM_TRIALS} 轮 box_world 随机目标追踪实验...")
        total_start_time = time.perf_counter()

        for i in range(1, NUM_TRIALS + 1):
            round_start_time = time.perf_counter()
            box_world.clear_all_world_models()
            delete_all_goals()

            box_size_x, box_size_y = box_world.sample_safe_box_size()
            robot_x, robot_y, robot_yaw = box_world.sample_safe_robot_pose(box_size_x, box_size_y)
            robot_clearance_to_box, robot_clearance_to_wall = box_world._robot_clearance_metrics_at_pose(
                robot_x,
                robot_y,
                box_size_x,
                box_size_y,
            )

            box_world.spawn_box_world_obstacle(
                sdf_template_path=sdf_template_path,
                size_x=box_size_x,
                size_y=box_size_y,
                model_name=box_world.BOX_MODEL_NAME,
            )
            box_world.spawn_robot_entity(robot_x, robot_y, robot_yaw)

            random_x, random_y = box_world.sample_valid_goal(box_size_x, box_size_y, robot_x, robot_y)
            goal_name = f"box_goal_{i}"

            print(
                f"\n--- 第{i}/{NUM_TRIALS}轮 ---",
                f"\n{model_name}",
                "\nBOX WORLD",
                f"\nrobot ({robot_x:.2f}, {robot_y:.2f}, yaw={robot_yaw:.2f})",
                f"\n{goal_name} ({random_x:.2f}, {random_y:.2f})",
                f"\nbox_size=({box_size_x:.2f}, {box_size_y:.2f})",
            )

            spawn_goal_point(random_x, random_y, 0.2, name=goal_name)

            reached = track_single_goal(
                model_name=model_name,
                goal_xy=(random_x, random_y),
                recorder=recorder,
                goal_name=goal_name,
                timeout_sec=TIMEOUT_PER_GOAL,
                reach_threshold_m=0.3,
                collision_fail_distance_m=box_world.COLLISION_FAIL_DISTANCE_M,
                scene_data={
                    "trial_id": i,
                    "box_size_x": round(box_size_x, 4),
                    "box_size_y": round(box_size_y, 4),
                    "box_yaw_rad": box_world.BOX_YAW_RAD,
                    "robot_start_x": round(robot_x, 4),
                    "robot_start_y": round(robot_y, 4),
                    "robot_start_yaw": round(robot_yaw, 4),
                    "box_center_x": box_world.WORLD_CENTER_X,
                    "box_center_y": box_world.WORLD_CENTER_Y,
                    "world_center_x": box_world.WORLD_CENTER_X,
                    "world_center_y": box_world.WORLD_CENTER_Y,
                    "world_l1_radius": box_world.WORLD_L1_RADIUS,
                    "world_margin": box_world.WORLD_MARGIN,
                    "collision_fail_distance_m": box_world.COLLISION_FAIL_DISTANCE_M,
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
                f"本轮={box_world._format_seconds(round_elapsed)} | "
                f"累计={box_world._format_seconds(total_elapsed)} | "
            )

        print(f"\n所有实验已完成。数据记录在 '{os.path.join(output_dir, file_name)}' 中。\n")

    finally:
        try:
            box_world.clear_all_world_models()
            delete_all_goals()
        except Exception as exc:
            print(f"收尾清理时出现异常: {exc}\n")
        rclpy.shutdown()


if __name__ == "__main__":
    StartTest_BoxWorld_FiveModels("ppo_mixed_simple_fc_sft_epoch_020")
