# from test1_100_goals_empty_world import StartTest_EmptyWorld
# from test2_100_goals_box_world import StartTest_BoxWorld
# from test3_100_goals_door_world import StartTest_DoorWorld

# NUM_TRIALS_PER_TEST = 1

# Legacy Models (3)
# StartTest_EmptyWorld("Model_1_PPO_Ckpt_Step_10000", NUM_TRIALS=NUM_TRIALS_PER_TEST)
# StartTest_BoxWorld("Model_1_PPO_Ckpt_Step_10000", NUM_TRIALS=NUM_TRIALS_PER_TEST)
# StartTest_DoorWorld("Model_1_PPO_Ckpt_Step_10000", NUM_TRIALS=NUM_TRIALS_PER_TEST)

# StartTest_EmptyWorld("Model_2_Supervised_Ckpt_Step_200000", NUM_TRIALS=NUM_TRIALS_PER_TEST)
# StartTest_BoxWorld("Model_2_Supervised_Ckpt_Step_200000", NUM_TRIALS=NUM_TRIALS_PER_TEST)
# StartTest_DoorWorld("Model_2_Supervised_Ckpt_Step_200000", NUM_TRIALS=NUM_TRIALS_PER_TEST)

# StartTest_EmptyWorld("Model_3_PPO_Ckpt_Step_740000", NUM_TRIALS=NUM_TRIALS_PER_TEST)
# StartTest_BoxWorld("Model_3_PPO_Ckpt_Step_740000", NUM_TRIALS=NUM_TRIALS_PER_TEST)
# StartTest_DoorWorld("Model_3_PPO_Ckpt_Step_740000", NUM_TRIALS=NUM_TRIALS_PER_TEST)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from test_empty_world_5_models import StartTest_EmptyWorld
from test_box_world_5_models import StartTest_BoxWorld
from test_door_world_5_models import StartTest_DoorWorld


NUM_TRIALS_PER_TEST = 400

StartTest_EmptyWorld("simple_fc_sft", NUM_TRIALS=NUM_TRIALS_PER_TEST)         # 空环境不会动原地打转
StartTest_BoxWorld("simple_fc_sft", NUM_TRIALS=NUM_TRIALS_PER_TEST)           # Box不会动原地打转
StartTest_DoorWorld("simple_fc_sft", NUM_TRIALS=NUM_TRIALS_PER_TEST)          # Door不会动原地打转

# StartTest_EmptyWorld("cnn_lstm_sft_nodoor", NUM_TRIALS=NUM_TRIALS_PER_TEST)   # 空环境非常慢
# StartTest_BoxWorld("cnn_lstm_sft_nodoor", NUM_TRIALS=NUM_TRIALS_PER_TEST)     # Box不会动原地打转
# StartTest_DoorWorld("cnn_lstm_sft_nodoor", NUM_TRIALS=NUM_TRIALS_PER_TEST)    # Door不会动原地打转

# StartTest_EmptyWorld("cnn_lstm_sft", NUM_TRIALS=NUM_TRIALS_PER_TEST)          # 可以动
# StartTest_BoxWorld("cnn_lstm_sft", NUM_TRIALS=NUM_TRIALS_PER_TEST)            # Box动不了，打转
# StartTest_DoorWorld("cnn_lstm_sft", NUM_TRIALS=NUM_TRIALS_PER_TEST)           # Door正常

# StartTest_EmptyWorld("cnn_gru_sft", NUM_TRIALS=NUM_TRIALS_PER_TEST)           # 空环境可以动
# StartTest_BoxWorld("cnn_gru_sft", NUM_TRIALS=NUM_TRIALS_PER_TEST)             # Box容易卡住
# StartTest_DoorWorld("cnn_gru_sft", NUM_TRIALS=NUM_TRIALS_PER_TEST)            # Door容易卡住

# StartTest_EmptyWorld("fc_lstm_sft", NUM_TRIALS=NUM_TRIALS_PER_TEST)           # 空环境不会动
# StartTest_BoxWorld("fc_lstm_sft", NUM_TRIALS=NUM_TRIALS_PER_TEST)             # Box不会动原地打转
# StartTest_DoorWorld("fc_lstm_sft", NUM_TRIALS=NUM_TRIALS_PER_TEST)            # Door很慢