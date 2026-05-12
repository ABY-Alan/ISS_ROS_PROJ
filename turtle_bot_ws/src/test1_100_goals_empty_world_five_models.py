import os
import random
import time

import rclpy

from chase_goal_record_data_PpoMixedFiveModels import (
    DataRecorder,
    get_output_tag,
    get_supported_model_names,
    track_single_goal,
)
from gazebo_goal_point import delete_all_goals, spawn_goal_point


MAP_SIZE_X = 5.0
MAP_SIZE_Y = 5.0
TIMEOUT_PER_GOAL = 300.0


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


def StartTest_EmptyWorld_FiveModels(model_name: str, NUM_TRIALS: int = 200):
    if model_name not in get_supported_model_names():
        supported = ", ".join(get_supported_model_names())
        raise ValueError(f"不支持的模型名: {model_name}。可选值: {supported}")

    base_file_name = f"Output_{get_output_tag(model_name)}_Empty_World_Test.csv"

    rclpy.init()

    delete_all_goals()

    output_dir = os.path.join(os.path.dirname(__file__), "Models", "Outputs")
    os.makedirs(output_dir, exist_ok=True)
    file_name = get_unique_filename(output_dir, base_file_name)
    recorder = DataRecorder(filename=os.path.join(output_dir, file_name))

    print(f"开始进行 {NUM_TRIALS} 轮随机目标追踪实验...")
    total_start_time = time.perf_counter()

    try:
        for i in range(1, NUM_TRIALS + 1):
            round_start_time = time.perf_counter()

            random_x = random.uniform(-MAP_SIZE_X, MAP_SIZE_X)
            random_y = random.uniform(-MAP_SIZE_Y, MAP_SIZE_Y)
            goal_name = f"random_goal_{i}"

            print(
                f"\n--- 第{i}/{NUM_TRIALS}轮 ---",
                f"\n{model_name}",
                "\nEMPTY WORLD",
                f"\n{goal_name} ({random_x:.2f}, {random_y:.2f})",
            )

            spawn_goal_point(random_x, random_y, 0.2, name=goal_name)

            reached = track_single_goal(
                model_name=model_name,
                goal_xy=(random_x, random_y),
                recorder=recorder,
                goal_name=goal_name,
                timeout_sec=TIMEOUT_PER_GOAL,
                reach_threshold_m=0.3,
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
        rclpy.shutdown()


if __name__ == "__main__":
    StartTest_EmptyWorld_FiveModels("ppo_mixed_simple_fc_sft_epoch_020")
