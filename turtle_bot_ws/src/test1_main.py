from test1_100_goals_empty_world import StartTest_EmptyWorld
from test2_100_goals_box_world import StartTest_BoxWorld
from test3_100_goals_door_world import StartTest_DoorWorld

from test1_100_goals_empty_world_five_models import StartTest_EmptyWorld_FiveModels
from test2_100_goals_box_world_five_models import StartTest_BoxWorld_FiveModels
from test3_100_goals_door_world_five_models import StartTest_DoorWorld_FiveModels


NUM_TRIALS_PER_TEST = 200

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


# # New Mixed PPO Models (5)
StartTest_EmptyWorld_FiveModels("ppo_mixed_simple_fc_sft_epoch_020", NUM_TRIALS=NUM_TRIALS_PER_TEST)
# StartTest_BoxWorld_FiveModels("ppo_mixed_simple_fc_sft_epoch_020", NUM_TRIALS=NUM_TRIALS_PER_TEST)
# StartTest_DoorWorld_FiveModels("ppo_mixed_simple_fc_sft_epoch_020", NUM_TRIALS=NUM_TRIALS_PER_TEST)

# StartTest_EmptyWorld_FiveModels("ppo_mixed_cnn_lstm_sft_nodoor_epoch_020", NUM_TRIALS=NUM_TRIALS_PER_TEST)
# StartTest_BoxWorld_FiveModels("ppo_mixed_cnn_lstm_sft_nodoor_epoch_020", NUM_TRIALS=NUM_TRIALS_PER_TEST)
# StartTest_DoorWorld_FiveModels("ppo_mixed_cnn_lstm_sft_nodoor_epoch_020", NUM_TRIALS=NUM_TRIALS_PER_TEST)

# StartTest_EmptyWorld_FiveModels("ppo_mixed_cnn_lstm_sft_epoch_020", NUM_TRIALS=NUM_TRIALS_PER_TEST)
# StartTest_BoxWorld_FiveModels("ppo_mixed_cnn_lstm_sft_epoch_020", NUM_TRIALS=NUM_TRIALS_PER_TEST)
# StartTest_DoorWorld_FiveModels("ppo_mixed_cnn_lstm_sft_epoch_020", NUM_TRIALS=NUM_TRIALS_PER_TEST)

# StartTest_EmptyWorld_FiveModels("ppo_mixed_cnn_gru_sft_epoch_020", NUM_TRIALS=NUM_TRIALS_PER_TEST)
# StartTest_BoxWorld_FiveModels("ppo_mixed_cnn_gru_sft_epoch_020", NUM_TRIALS=NUM_TRIALS_PER_TEST)
# StartTest_DoorWorld_FiveModels("ppo_mixed_cnn_gru_sft_epoch_020", NUM_TRIALS=NUM_TRIALS_PER_TEST)

# StartTest_EmptyWorld_FiveModels("ppo_mixed_fc_lstm_sft_epoch_020", NUM_TRIALS=NUM_TRIALS_PER_TEST)
# StartTest_BoxWorld_FiveModels("ppo_mixed_fc_lstm_sft_epoch_020", NUM_TRIALS=NUM_TRIALS_PER_TEST)
# StartTest_DoorWorld_FiveModels("ppo_mixed_fc_lstm_sft_epoch_020", NUM_TRIALS=NUM_TRIALS_PER_TEST)
