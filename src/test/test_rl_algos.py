'''
|1|2|3 |4 |
|5|X|6 |7 | 7 -> -1
|8|9|10|11| 11 -> +1
'''
from src.rl_algos.base_rl_algo import Policy
from src.rl_algos.control import SarsaRL, SarsaLambdaRL, QLearningTabular
from src.rl_algos.planning import MCTabularRL, TDZTabularRL, TDLambdaTabularRL
import numpy as np

starting_distribution = [(1, 1)]

base_policy = Policy({
    1: {'r': 0.5, 'd': 0.5},
    2: {'r': 1},
    3: {'r': 0.5, 'd': 0.5},
    4: {'d': 1},
    5: {'d': 1},
    6: {'r': 1},
    7: {'s': 1},
    8: {'r': 1},
    9: {'r': 1},
    10: {'r': 1},
    11: {'s': 1},
})

states = list(range(1, 12))
terminal_states = {7, 11}
actions = {1: ['r', 'd'], 2: ['l', 'r'], 3: ['l', 'r', 'd'],
           4: ['l', 'd'], 5: ['u', 'd'], 6: ['r', 'u', 'd'],
           7: ['s'], 8: ['u', 'r'], 9: ['l', 'r'], 10: ['l', 'r', 'u'],
           11: ['s']}
mdp_transitions = {
    (1, 'r'): lambda: (2, 0),
    (1, 'd'): lambda: (5, 0),

    (2, 'r'): lambda: (3, 0),
    (2, 'l'): lambda: (1, 0),

    (3, 'l'): lambda: (2, 0),
    (3, 'r'): lambda: (4, 0),
    (3, 'd'): lambda: (6, 0),

    (4, 'l'): lambda: (3, 0),
    (4, 'd'): lambda: (7, -1),

    (5, 'u'): lambda: (1, 0),
    (5, 'd'): lambda: (8, 0),

    (6, 'u'): lambda: (3, 0),
    (6, 'd'): lambda: (10, 0),
    (6, 'r'): lambda: (7, -1),

    (7, 's'): lambda: (7, 0),

    (8, 'u'): lambda: (5, 0),
    (8, 'r'): lambda: (9, 0),

    (9, 'l'): lambda: (8, 0),
    (9, 'r'): lambda: (10, 0),

    (10, 'l'): lambda: (9, 0),
    (10, 'r'): lambda: (11, 1),
    (10, 'u'): lambda: (6, 0),

    (11, 's'): lambda: (11, 0)
}


def test_mc_tabular():
    np.random.seed(0)
    rl_mc = MCTabularRL(mdp_transitions, terminal_states=terminal_states,
                        state_actions=actions, gamma=0.9, num_episodes=10, max_iter=100,
                        starting_distribution=starting_distribution)
    vf = rl_mc.get_value_function(base_policy)
    assert vf == {1: 0.10206000000000008,
                  2: -0.81,
                  3: -0.9,
                  4: -1.0,
                  5: 0.7290000000000001,
                  6: -1.0,
                  7: 0.0,
                  8: 0.8100000000000002,
                  9: 0.9,
                  10: 1.0,
                  11: 0.0}


def test_tdzero():
    np.random.seed(0)
    rl_mc = TDZTabularRL(mdp_transitions, terminal_states=terminal_states,
                         state_actions=actions, gamma=0.9, num_episodes=10000, max_iter=100,
                         starting_distribution=starting_distribution)

    vf = rl_mc.get_value_function(base_policy)
    assert vf == {1: 0.0463436243563566,
                  2: -0.8099999999521277,
                  3: -0.8999999999729613,
                  4: -0.9999999999899842,
                  5: 0.728999999999981,
                  6: -0.9999999999813234,
                  7: 0.0,
                  8: 0.809999999999985,
                  9: 0.8999999999999895,
                  10: 0.9999999999999944,
                  11: 0.0}


def test_td_lmbda():
    np.random.seed(0)
    rl_mc = TDLambdaTabularRL(mdp_transitions, terminal_states=terminal_states,
                              state_actions=actions, gamma=0.9, num_episodes=1000, max_iter=100,
                              starting_distribution=starting_distribution)
    vf = rl_mc.get_value_function(base_policy, lmbda=0.001)
    assert vf == {1: 0.006341347398420489,
                  2: -0.587828929977394,
                  3: -0.758057809198005,
                  4: -0.9288694479745844,
                  5:  0.5371648418853612,
                  6: -0.9076278356441421,
                  7: 0.0,
                  8: 0.7102080081871613,
                  9: 0.8642539175086417,
                  10: 0.9934295169575857,
                  11: 0.0}


def test_sarsa():
    np.random.seed(0)
    rl_srsa = SarsaRL(mdp_transitions, terminal_states=terminal_states,
                      state_actions=actions, gamma=0.9, num_episodes=5000, max_iter=100,
                      starting_distribution=starting_distribution)

    q_func = rl_srsa.learn_q_value_function(alpha=0.01)
    policy = rl_srsa.get_epsilon_greedy_policy(q_func, 0)
    p = {state: policy.get_action(state) for state in rl_srsa.states}
    expected_policy = {1: 'd', 2: 'l', 3: 'd', 4: 'l', 5: 'd', 6: 'd', 7: 's', 8: 'r', 9: 'r', 10: 'r', 11: 's'}
    assert p == expected_policy


def test_sarsa_lambda():
    rl_srsaL = SarsaLambdaRL(mdp_transitions, terminal_states=terminal_states,
                             state_actions=actions, gamma=0.9, num_episodes=10000, max_iter=100,
                             starting_distribution=starting_distribution)

    q_func = rl_srsaL.learn_q_value_function(alpha=0.01, lmbda=0.5)
    policy = rl_srsaL.get_epsilon_greedy_policy(q_func, 0)
    p = {state: policy.get_action(state) for state in rl_srsaL.states}
    expected_policy = {1: 'd', 2: 'l', 3: 'd', 4: 'l', 5: 'd', 6: 'd', 7: 's', 8: 'r', 9: 'r', 10: 'r', 11: 's'}
    assert p == expected_policy


def test_ql_tab():
    rl_ql = QLearningTabular(mdp_transitions, terminal_states=terminal_states,
                             state_actions=actions, gamma=0.9, num_episodes=1000, max_iter=100,
                             starting_distribution=starting_distribution)
    q_func = rl_ql.learn_q_value_function(alpha=0.01)
    policy = rl_ql.get_epsilon_greedy_policy(q_func, 0)
    p = {state: policy.get_action(state) for state in rl_ql.states}
    expected_policy = {1: 'r', 2: 'r', 3: 'd', 4: 'l', 5: 'd', 6: 'd', 7: 's', 8: 'r', 9: 'r', 10: 'r', 11: 's'}
    assert p == expected_policy
