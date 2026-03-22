from test1_100_goals_empty_world import StartTest_EmptyWorld
from test2_100_goals_box_world import StartTest_BoxWorld
from test3_100_goals_door_world import StartTest_DoorWorld

NUM_TRIALS_PER_TEST = 400

# # Test Model 1
# StartTest_EmptyWorld("Model_1_PPO_Ckpt_Step_10000", NUM_TRIALS=NUM_TRIALS_PER_TEST)
# StartTest_BoxWorld("Model_1_PPO_Ckpt_Step_10000", NUM_TRIALS=NUM_TRIALS_PER_TEST)
# StartTest_DoorWorld("Model_1_PPO_Ckpt_Step_10000", NUM_TRIALS=NUM_TRIALS_PER_TEST)

StartTest_EmptyWorld("Model_2_Supervised_Ckpt_Step_200000", NUM_TRIALS=NUM_TRIALS_PER_TEST)
StartTest_BoxWorld("Model_2_Supervised_Ckpt_Step_200000", NUM_TRIALS=NUM_TRIALS_PER_TEST)
StartTest_DoorWorld("Model_2_Supervised_Ckpt_Step_200000", NUM_TRIALS=NUM_TRIALS_PER_TEST)

StartTest_EmptyWorld("Model_3_PPO_Ckpt_Step_740000", NUM_TRIALS=NUM_TRIALS_PER_TEST)
StartTest_BoxWorld("Model_3_PPO_Ckpt_Step_740000", NUM_TRIALS=NUM_TRIALS_PER_TEST)
StartTest_DoorWorld("Model_3_PPO_Ckpt_Step_740000", NUM_TRIALS=NUM_TRIALS_PER_TEST)
