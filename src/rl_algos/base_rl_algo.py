import numpy as np
from typing import Sequence, Mapping, Tuple, Set
from src.utils.generic_typevars import S, A
from src.utils.typevars import Tab_RL_Transitions


class Policy():
    def __init__(self, policy_info: Mapping[S, Mapping[A, float]]):
        self.policy_info = {state: [(action, prob) for action, prob in policy_info[state].items()] for state
                            in policy_info.keys()}

    def get_action(self, state):
        if state not in self.policy_info.keys():
            assert False, f'attempting to build an invalid shit, {state} not in states'
        return np.random.choice([action for action, _ in self.policy_info[state]], 1,
                                p=[prob for _, prob in self.policy_info[state]])[0]


class BaseTabularRL():

    def __init__(self, transitions: Tab_RL_Transitions, terminal_states: Set[S],
                 state_actions: Mapping[S, Set[A]], gamma: float, num_episodes=100, max_iter=100,
                 starting_distribution: Set[Tuple[S, float]] = None):
        self.gamma = gamma
        self.states = state_actions.keys()
        self.transitions = transitions
        self.terminal_states = terminal_states
        self.state_actions = state_actions
        self.num_episodes = num_episodes
        self.max_iter = max_iter
        if starting_distribution is None:
            self.starting_distribution = [(s, 1 / len(self.states)) for s in self.states]
        else:
            self.starting_distribution = starting_distribution

    def run_episode(self, starting_state: S, policy: Policy) -> Sequence[Tuple[S, A, float, bool]]:
        visited = set()
        curr_state = starting_state
        iteration = 0
        continue_iter = True
        episode = []

        while continue_iter:
            iteration += 1
            first_visit = curr_state not in visited
            action = policy.get_action(curr_state)
            next_state, reward = self.transitions[(curr_state, action)]()
            episode.append((curr_state, action, reward, first_visit))

            if iteration >= self.max_iter or curr_state in self.terminal_states:
                continue_iter = False
            curr_state = next_state
        return episode

    def get_epsilon_greedy_policy(self, q_value_function, epsilon):
        policy_info = {}
        for state in self.states:
            actions = self.state_actions[state]
            best_action = max([(q_value_function[state, a], a) for a in actions])[1]
            policy = {a: epsilon / len(actions) for a in actions}
            policy[best_action] += 1 - epsilon
            policy_info[state] = policy
        return Policy(policy_info)

    def get_value_function_from_policy(self, policy: Policy):
        pass

    def get_value_function(self, policy: Policy) -> Mapping[S, float]:
        pass

    def get_q_value_function(self, policy: Policy) -> Mapping[Tuple[S, A], float]:
        pass
