from src.processes.markov_reward_process import BaseMarkovRewardProcessImpl


def test_get_value_empty():
    rewards = {
        1: 7.0,
        2: 10.0,
        3: 0.0
    }
    transitions = {
        1: {1: 0.6, 2: 0.3, 3: 0.1},
        2: {1: 0.1, 2: 0.2, 3: 0.7},
        3: {3: 1.0}
    }

    mp = BaseMarkovRewardProcessImpl(transitions, rewards, 1.0)
    value_function = mp.get_value_func_vec()
    print(value_function)
    value_function = mp.get_value_func()
    print(value_function)
