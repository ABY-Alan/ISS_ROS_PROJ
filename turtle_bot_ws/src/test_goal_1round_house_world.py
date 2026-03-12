# # from gazebo_goal_point import spawn_goal_point, delete_all_goals
# from gazebo_goal_point import spawn_goal_point, delete_all_goals
# from chase_goal_record_data import track_single_goal

# delete_all_goals()

# goal_1 = spawn_goal_point(1.0, -1.0, 0.2, name="goal_1")
# goal_2 = spawn_goal_point(1.0, 0.5, 0.2, name="goal_2")
# goal_3 = spawn_goal_point(-4.5, 0.5, 0.2, name="goal_3")
# goal_4 = spawn_goal_point(-4.5, 3.5, 0.2, name="goal_4")
# goal_5 = spawn_goal_point(-6.5, 3.5, 0.2, name="goal_5")
# goal_6 = spawn_goal_point(-6.5, -3.0, 0.2, name="goal_6")

# # goals = [goal_1, goal_2, goal_3, goal_4, goal_5, goal_6]
# goals = [goal_1]
# for goal in goals:
#     reached = track_single_goal(
#         goal_xy = goal,
#         reach_threshold_m = 0.25,
#         timeout_sec = 60.0,
#         control_rate_hz = 10.0,
#     )


# --- 修改后的 test_goal.py ---
from gazebo_goal_point import spawn_goal_point, delete_all_goals
from chase_goal_record_data import track_single_goal, DataRecorder # 导入 DataRecorder

delete_all_goals()

# 1. 在循环外初始化一个全局记录器
global_recorder = DataRecorder(filename="combined_experiment_results.csv")

goals_data = [
    {"pos": (1.0, -1.0), "name": "goal_1"},
    {"pos": (1.0, 0.5), "name": "goal_2"},
    {"pos": (-4.5, 0.5), "name": "goal_3"},
    {"pos": (-4.5, 3.5), "name": "goal_4"},
    {"pos": (-6.5, 3.5), "name": "goal_5"},
    {"pos": (-6.5, -3.0), "name": "goal_6"},
]

for g in goals_data:
    spawn_goal_point(g["pos"][0], g["pos"][1], 0.2, name=g["name"])
    
    print(f"开始追踪: {g['name']}")
    # 2. 将记录器实例传入每个追踪任务
    reached = track_single_goal(
        goal_xy = g["pos"],
        recorder = global_recorder, # 传入记录器
        goal_name = g["name"],
        reach_threshold_m = 0.25,
        timeout_sec = 60.0,
        control_rate_hz = 50.0,
    )
    
delete_all_goals()
print("所有目标追踪完毕，数据已合并。")