from typing import Mapping

from src.processes.markov_decision_process import MarkovDecisionProcess
from src.utils.func_utils import eq_to_epsilon
from src.utils.generic_typevars import S, A
from src.utils.typevars import MDPActions, MDPTransitions


def policy_iteration(mdp: MarkovDecisionProcess) -> Mapping[S, A]:
    actions: MDPActions = mdp.get_actions_by_states()
    base_policy = {s: {(a[0], 1)} for s, a in actions.items()}
    value_function_for_policy = mdp.get_mrp(base_policy).get_value_func()
    greedy_policy = get_greedy_policy(mdp, value_function_for_policy)
    while not check_policy_equivalence(base_policy, greedy_policy):
        base_policy = greedy_policy
        value_function_for_policy = mdp.get_mrp(base_policy).get_value_func()
        greedy_policy = get_greedy_policy(mdp, value_function_for_policy)
    print('vf after policy is determined', value_function_for_policy)
    return greedy_policy


def value_iteration(mdp: MarkovDecisionProcess) -> Mapping[S, float]:
    base_value_function = {s: 0 for s in mdp.get_actions_by_states().keys()}
    next_value_function = iterate_on_value_function(mdp, base_value_function)
    while not check_value_fuction_equivalence(base_value_function, next_value_function):
        base_value_function = next_value_function
        next_value_function = iterate_on_value_function(mdp, base_value_function)
    return base_value_function


def iterate_on_value_function(mdp: MarkovDecisionProcess, base_vf: Mapping[S, float]) -> Mapping[S, float]:
    actions: MDPActions = mdp.get_actions_by_states()
    new_vf = {}
    for s in actions.keys():
        action_values = [(action, extract_value_of_action(mdp, action, s, base_vf)) for action in actions[s]]
        best_action_reward = max([x[1] for x in action_values])
        new_vf[s] = best_action_reward
    return new_vf


def extract_value_of_action(mdp: MarkovDecisionProcess, action: A, state: S, value_function):
    transitions: MDPTransitions = mdp.get_state_action_transitions()
    rewards = mdp.get_rewards()
    discount = mdp.get_discount()
    return rewards[state][action] + discount * sum([p * value_function[s_prime]
                                                    for s_prime, p in
                                                    transitions[state][action].items()])


def check_value_fuction_equivalence(v1, v2, epsilon=1e-8) -> bool:
    assert v1.keys() == v2.keys(), "comparing policies with different state spaces"
    for state in v1:
        if not eq_to_epsilon(v1[state], v2[state], epsilon):
            return False
    return True


def check_policy_equivalence(p1, p2) -> bool:
    assert p1.keys() == p2.keys(), "comparing policies with different state spaces"
    for state in p1:
        if p1[state] != p2[state]:
            return False
    return True


def get_greedy_policy(mdp: MarkovDecisionProcess, value_function: Mapping[S, float]) -> Mapping[S, A]:
    actions: MDPActions = mdp.get_actions_by_states()
    policy = {}
    for s in mdp.get_non_terminal_states():
        actions_rewards = {}
        for action in actions[s]:
            actions_rewards[action] = extract_value_of_action(mdp, action, s, value_function)
        policy[s] = {(max(actions_rewards, key=actions_rewards.get), 1)}
    for s in mdp.get_terminal_states():
        policy[s] = {(actions[s][0], 1)}
    return policy
