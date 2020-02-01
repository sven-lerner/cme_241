from src.control.iterative_methods import policy_iteration, value_iteration, check_value_fuction_equivalence, \
    get_greedy_policy, check_policy_equivalence
from src.processes.markov_decision_process import BaseMarkovDecisionProcessImpl


def test_basic_mrp_policy_iteration():
    transitions = {
        1: {
            'a': {1: 0.3, 2: 0.6, 3: 0.1},
            'b': {2: 0.3, 3: 0.7},
            'c': {1: 0.2, 2: 0.4, 3: 0.4}
        },
        2: {
            'a': {1: 0.3, 2: 0.6, 3: 0.1},
            'c': {1: 0.2, 2: 0.4, 3: 0.4}
        },
        3: {
            'a': {3: 1.0},
            'b': {3: 1.0}
        }
    }

    rewards = {
        1: {'a': 5.0, 'b': 2.8, 'c': -7.2},
        2: {'a': 5.0, 'c': -7.2},
        3: {'a': 0.0, 'b': 0.0}
    }
    terminal_states = {3}
    gamma = 0.95

    mdp = BaseMarkovDecisionProcessImpl(transitions, rewards, terminal_states, gamma)
    policy = policy_iteration(mdp)
    assert policy == {1: {('a', 1)}, 2: {('a', 1)}, 3: {('a', 1)}}


def test_basic_mrp_value_iteration():
    transitions = {
        1: {
            'a': {1: 0.3, 2: 0.6, 3: 0.1},
            'b': {2: 0.3, 3: 0.7},
            'c': {1: 0.2, 2: 0.4, 3: 0.4}
        },
        2: {
            'a': {1: 0.3, 2: 0.6, 3: 0.1},
            'c': {1: 0.2, 2: 0.4, 3: 0.4}
        },
        3: {
            'a': {3: 1.0},
            'b': {3: 1.0}
        }
    }

    rewards = {
        1: {'a': 5.0, 'b': 2.8, 'c': -7.2},
        2: {'a': 5.0, 'c': -7.2},
        3: {'a': 0.0, 'b': 0.0}
    }
    terminal_states = {3}
    gamma = 0.95

    mdp = BaseMarkovDecisionProcessImpl(transitions, rewards, terminal_states, gamma)
    value_function = value_iteration(mdp)
    assert check_value_fuction_equivalence(value_function,
                                            {1: 34.48275855319572,
                                            2: 34.48275855319572,
                                            3: 0})

def test_value_and_policy_iteration_converge():
    transitions = {
        1: {
            'a': {1: 0.3, 2: 0.6, 3: 0.1},
            'b': {2: 0.3, 3: 0.7},
            'c': {1: 0.2, 2: 0.4, 3: 0.4}
        },
        2: {
            'a': {1: 0.3, 2: 0.6, 3: 0.1},
            'c': {1: 0.2, 2: 0.4, 3: 0.4}
        },
        3: {
            'a': {3: 1.0},
            'b': {3: 1.0}
        }
    }

    rewards = {
        1: {'a': 5.0, 'b': 2.8, 'c': -7.2},
        2: {'a': 5.0, 'c': -7.2},
        3: {'a': 0.0, 'b': 0.0}
    }
    terminal_states = {3}
    gamma = 0.95

    mdp = BaseMarkovDecisionProcessImpl(transitions, rewards, terminal_states, gamma)
    policy = policy_iteration(mdp)
    value_fn = value_iteration(mdp)
    mrp = mdp.get_mrp(policy)

    mrp_value_function = mrp.get_value_func()

    assert check_value_fuction_equivalence(mrp_value_function,
                                           value_fn, 1e-06)
    assert check_policy_equivalence(get_greedy_policy(mdp, value_fn),
                                    policy)
