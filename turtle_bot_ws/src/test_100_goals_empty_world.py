import rclpy
from gazebo_goal_point import spawn_goal_point, delete_all_goals
from chase_goal_record_data import track_single_goal, DataRecorder
import random
import time
import os

# --- 配置参数 ---
NUM_TRIALS = 100
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

def main():
    rclpy.init()
    
    # 1. 初始清理
    delete_all_goals()
    
    # 2. 初始化全局记录器
    base_file_name = "Output_PPO_Ckpt_Empty_World_Test_100.csv"
    output_dir = os.path.join(os.path.dirname(__file__), "Models", "Outputs")
    os.makedirs(output_dir, exist_ok=True)
    file_name = get_unique_filename(output_dir, base_file_name)
    recorder = DataRecorder(filename=os.path.join(output_dir, file_name))
    
    print(f"开始进行 {NUM_TRIALS} 轮随机目标追踪实验...")
    
    for i in range(1, NUM_TRIALS + 1):
        # 3. 随机生成目标点坐标
        random_x = random.uniform(-MAP_SIZE_X, MAP_SIZE_X)
        random_y = random.uniform(-MAP_SIZE_Y, MAP_SIZE_Y)
        goal_name = f"random_goal_{i}"
        
        print(f"\n--- 第 {i}/{NUM_TRIALS} 轮: 目标 {goal_name} ({random_x:.2f}, {random_y:.2f}) ---")
        
        # 4. 生成目标点并在Gazebo中显示
        spawn_goal_point(random_x, random_y, 0.2, name=goal_name)
        
        # 5. 追踪目标
        reached = track_single_goal(
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
        
        # 7. 可选：在下一轮前让机器人休息一下
        # time.sleep(1)

    print(f"\n所有实验已完成。数据记录在 '{os.path.join(output_dir, file_name)}' 中。")
    rclpy.shutdown()

if __name__ == "__main__":
    main()