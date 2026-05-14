#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import time
from typing import Any, Callable

import rclpy

from gazebo_goal_point import spawn_goal_point, delete_all_goals
from chase_goal_record_data_5_models import DataRecorder, track_single_goal


MAP_SIZE_X = 5.0
MAP_SIZE_Y = 5.0
TIMEOUT_PER_GOAL = 300.0
COLLISION_FAIL_DISTANCE_M = 0.30

MODEL_NAMES = [
    "simple_fc_sft",
    "cnn_lstm_sft_nodoor",
    "cnn_lstm_sft",
    "cnn_gru_sft",
    "fc_lstm_sft",
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


def _parse_track_result(result: Any) -> bool:
    if isinstance(result, dict):
        return bool(result.get("success", False))
    return bool(result)


def _get_status(result: Any) -> str:
    if isinstance(result, dict):
        return str(result.get("status", "failed"))
    return "success" if bool(result) else "failed"


def StartTest_EmptyWorld(ModelName: str, NUM_TRIALS: int = 200):
    track_single_goal_fn: Callable[..., dict] = track_single_goal
    DataRecorderCls: type = DataRecorder

    if not rclpy.ok():
        rclpy.init()

    script_dir = os.path.dirname(__file__)
    ckpt_dir = os.path.join(script_dir, "Models")
    output_dir = os.path.join(script_dir, "Models", "Outputs")
    os.makedirs(output_dir, exist_ok=True)

    base_file_name = f"Output_{ModelName}_Empty_World_Test.csv"
    file_name = get_unique_filename(output_dir, base_file_name)
    output_path = os.path.join(output_dir, file_name)

    recorder = DataRecorderCls(
        filename=output_path,
        extra_fields=[
            "trial_id",
            "model_name",
            "world_name",
            "collision_happened",
        ],
    )

    try:
        delete_all_goals()

        print(f"开始进行 {NUM_TRIALS} 轮 EmptyWorld 随机目标追踪实验...")
        print(f"当前模型: {ModelName}")
        print(f"输出文件: {output_path}")

        total_start_time = time.perf_counter()

        for i in range(1, NUM_TRIALS + 1):
            round_start_time = time.perf_counter()

            random_x = random.uniform(-MAP_SIZE_X, MAP_SIZE_X)
            random_y = random.uniform(-MAP_SIZE_Y, MAP_SIZE_Y)
            goal_name = f"empty_goal_{i}"

            print(
                f"\n--- 第{i}/{NUM_TRIALS}轮 ---"
                f"\n模型: {ModelName}"
                f"\n场景: EMPTY WORLD"
                f"\n目标: {goal_name} ({random_x:.2f}, {random_y:.2f})"
            )

            spawn_goal_point(random_x, random_y, 0.2, name=goal_name)

            result = track_single_goal_fn(
                goal_xy=(random_x, random_y),
                recorder=recorder,
                goal_name=goal_name,
                model_name=ModelName,
                ckpt_dir=ckpt_dir,
                timeout_sec=TIMEOUT_PER_GOAL,
                reach_threshold_m=0.3,
                collision_fail_distance_m=COLLISION_FAIL_DISTANCE_M,
                scene_data={
                    "trial_id": i,
                    "model_name": ModelName,
                    "world_name": "empty_world",
                    "collision_happened": 0,
                },
            )

            if _parse_track_result(result):
                print(f"成功: {goal_name} 已到达")
            else:
                print(f"失败: {goal_name} 未到达, status={_get_status(result)}")

            delete_all_goals()

            round_elapsed = time.perf_counter() - round_start_time
            total_elapsed = time.perf_counter() - total_start_time
            print(
                f"耗时统计: "
                f"本轮={_format_seconds(round_elapsed)} | "
                f"累计={_format_seconds(total_elapsed)}"
            )

        print(f"\n所有实验已完成。数据记录在 '{output_path}' 中。\n")

    finally:
        try:
            delete_all_goals()
        except Exception as e:
            print(f"收尾清理时出现异常: {e}")
        rclpy.shutdown()


if __name__ == "__main__":
    # 单个模型：
    StartTest_EmptyWorld("simple_fc_sft", NUM_TRIALS=200)

    # 批量跑 5 个模型时，注释上面一行，取消下面注释：
    # for model_name in MODEL_NAMES:
    #     StartTest_EmptyWorld(model_name, NUM_TRIALS=200)
