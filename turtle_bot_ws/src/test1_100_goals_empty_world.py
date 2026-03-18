import rclpy
from gazebo_goal_point import spawn_goal_point, delete_all_goals

import random
import time
import os
from typing import Any, Callable

# --- 配置参数 ---
MAP_SIZE_X = 5.0 # 地图X范围 [-5.0, 5.0]
MAP_SIZE_Y = 5.0 # 地图Y范围 [-5.0, 5.0]
TIMEOUT_PER_GOAL = 300.0 # 每轮最大时长，防止死循环


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

def StartTest_EmptyWorld(ModelName: str, NUM_TRIALS: int = 200):
    
    if ModelName == "Model_1_PPO_Ckpt_Step_10000":
        from chase_goal_record_data_Model1PpoCkptStep10000 import (
            track_single_goal as _track_single_goal,
            DataRecorder as _DataRecorder,
        )
        base_file_name = "Output_Model_1_PPO_Ckpt_Empty_World_Test_100.csv"
    elif ModelName == "Model_2_PPO_Ckpt_Step_10000":
        from chase_goal_record_data_PpoCkptStep10000 import (
            track_single_goal as _track_single_goal,
            DataRecorder as _DataRecorder,
        )
        base_file_name = "Output_Model_2_PPO_Ckpt_Empty_World_Test_100.csv"
    else:
        # default to Model 1 if unknown model name is provided
        from chase_goal_record_data_Model1PpoCkptStep10000 import (
            track_single_goal as _track_single_goal,
            DataRecorder as _DataRecorder,
        )
        base_file_name = "Output_Model_1_PPO_Ckpt_Empty_World_Test_100.csv"

    # 统一引用，避免分支导入后产生 Union 类型冲突。
    track_single_goal_fn: Callable[..., bool] = _track_single_goal
    DataRecorderCls: type[Any] = _DataRecorder
        
    rclpy.init()
    
    # 1. 初始清理
    delete_all_goals()
    
    # 2. 初始化全局记录器
    output_dir = os.path.join(os.path.dirname(__file__), "Models", "Outputs")
    os.makedirs(output_dir, exist_ok=True)
    file_name = get_unique_filename(output_dir, base_file_name)
    recorder = DataRecorderCls(filename=os.path.join(output_dir, file_name))
    
    print(f"开始进行 {NUM_TRIALS} 轮随机目标追踪实验...")
    total_start_time = time.perf_counter()

    for i in range(1, NUM_TRIALS + 1):
        round_start_time = time.perf_counter()

        # 3. 随机生成目标点坐标
        random_x = random.uniform(-MAP_SIZE_X, MAP_SIZE_X)
        random_y = random.uniform(-MAP_SIZE_Y, MAP_SIZE_Y)
        goal_name = f"random_goal_{i}"
        
        print(
            f"\n--- 第{i}/{NUM_TRIALS}轮 ---",
            f"\n{ModelName}",
            f"\nEMPTY WORLD",
            f"\n{goal_name} ({random_x:.2f}, {random_y:.2f})"
        )
        
        # 4. 生成目标点并在Gazebo中显示
        spawn_goal_point(random_x, random_y, 0.2, name=goal_name)
        
        # 5. 追踪目标
        reached = track_single_goal_fn(
            goal_xy = (random_x, random_y),
            recorder = recorder,
            goal_name = goal_name,
            timeout_sec = TIMEOUT_PER_GOAL,
            reach_threshold_m = 0.3 # 到达阈值
        )
        
        if reached:
            print(f"成功: {goal_name} 已到达")
        else:
            print(f"失败: {goal_name} 未到达")
            
        # 6. 删除当前目标点，准备下一轮
        delete_all_goals()

        round_elapsed = time.perf_counter() - round_start_time
        total_elapsed = time.perf_counter() - total_start_time

        print(
            f"耗时统计: "
            f"本轮={_format_seconds(round_elapsed)} | "
            f"累计={_format_seconds(total_elapsed)} | "
        )
        
    print(f"\n所有实验已完成。数据记录在 '{os.path.join(output_dir, file_name)}' 中。\n")
    rclpy.shutdown()

if __name__ == "__main__":
    # StartTest_EmptyWorld("Model_1_PPO_Ckpt_Step_10000")
    StartTest_EmptyWorld("Model_2_PPO_Ckpt_Step_10000")