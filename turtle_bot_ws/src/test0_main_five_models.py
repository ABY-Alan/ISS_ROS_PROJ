from chase_goal_record_data_PpoMixedFiveModels import get_supported_model_names
from test1_100_goals_empty_world_five_models import StartTest_EmptyWorld_FiveModels
from test2_100_goals_box_world_five_models import StartTest_BoxWorld_FiveModels
from test3_100_goals_door_world_five_models import StartTest_DoorWorld_FiveModels


NUM_TRIALS_PER_TEST = 200
MODEL_NAMES = list(get_supported_model_names())


def run_all_tests(num_trials: int = NUM_TRIALS_PER_TEST):
    for model_name in MODEL_NAMES:
        StartTest_EmptyWorld_FiveModels(model_name, NUM_TRIALS=num_trials)
        StartTest_BoxWorld_FiveModels(model_name, NUM_TRIALS=num_trials)
        StartTest_DoorWorld_FiveModels(model_name, NUM_TRIALS=num_trials)


# 单独跑某一个模型时，取消对应注释即可。
# StartTest_EmptyWorld_FiveModels("ppo_mixed_simple_fc_sft_epoch_020", NUM_TRIALS=NUM_TRIALS_PER_TEST)
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


if __name__ == "__main__":
    print("支持的五个模型:")
    for model_name in MODEL_NAMES:
        print(model_name)
    # run_all_tests()