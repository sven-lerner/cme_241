from src.processes.markov_reward_process import BaseMarkovRewardProcessImpl


def test_get_value_empty():
    mp = BaseMarkovRewardProcessImpl({1: {1: 1}, 2:{2:1}}, {1:1, 2:2}, 0.5)
    value_function = mp.get_value_func_vec()
    print(value_function)
    assert value_function == [1, 2]