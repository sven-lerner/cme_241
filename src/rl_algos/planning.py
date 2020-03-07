import numpy as np
from src.control.iterative_methods import policy_iteration
from src.processes.markov_decision_process import BaseMarkovDecisionProcessImpl
from typing import Callable, Sequence, Mapping, Tuple, Set

from src.rl_algos.base_rl_algo import BaseTabularRL, Policy
from src.utils.generic_typevars import S, A
from src.utils.typevars import Tab_RL_Transitions
from tqdm.notebook import tqdm
from collections import defaultdict


class MCTabularRL(BaseTabularRL):

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

    def get_value_function(self, policy: Policy, alpha=0.01, first_visit=False) -> Mapping[S, float]:
        value_function = {s: 0 for s in self.states}
        counts = {s: 0 for s in self.states}
        total_ret = {s: 0 for s in self.states}

        for _ in range(self.num_episodes):
            starting_state = np.random.choice([state for state, _ in self.starting_distribution], 1,
                                              p=[prob for _, prob in self.starting_distribution])[0]

            episode = self.run_episode(starting_state, policy)
            returns = len(episode) * [0]
            returns[-1] = episode[-1][-2]
            for i in range(len(episode) - 2, -1, -1):
                returns[i] = episode[i][-2] + self.gamma * returns[i + 1]

            for i, data in enumerate(episode):
                state, action, reward, first = data
                if first or not first_visit:
                    counts[state] += 1
                    total_ret[state] += returns[i]
        for state in self.states:
            value_function[state] = total_ret[state] / max(1, counts[state])
        return value_function


class TDZTabularRL(BaseTabularRL):

    def get_value_function(self, policy: Policy, alpha=0.01, first_visit=False) -> Mapping[S, float]:
        value_function = {s: 0 for s in self.states}
        counts = {s: 0 for s in self.states}
        total_ret = {s: 0 for s in self.states}
        print(self.num_episodes)
        for _ in range(self.num_episodes):
            starting_state = np.random.choice([state for state, _ in self.starting_distribution], 1,
                                              p=[prob for _, prob in self.starting_distribution])[0]

            episode = self.run_episode(starting_state, policy)

            for i, data in enumerate(episode[:-1]):
                state, action, reward, first = data
                next_state = episode[i + 1][0]
                value_function[state] += alpha * (
                            reward + self.gamma * value_function[next_state] - value_function[state])

        return value_function


class TDLambdaTabularRL(BaseTabularRL):

    def get_value_function(self, policy: Policy, alpha=0.01, first_visit=False,
                           lmbda: float = 0) -> Mapping[S, float]:
        print(self.states)
        value_function = {s: 0 for s in self.states}
        counts = {s: 0 for s in self.states}
        total_ret = {s: 0 for s in self.states}
        for _ in range(self.num_episodes):
            starting_state = np.random.choice([state for state, _ in self.starting_distribution], 1,
                                              p=[prob for _, prob in self.starting_distribution])[0]

            episode = self.run_episode(starting_state, policy)
            e_t = defaultdict(int)
            for i, data in enumerate(episode[:-1]):
                state, action, reward, first = data
                next_state = episode[i + 1][0]
                delta_t = reward + self.gamma * value_function[next_state] - value_function[state]
                e_t[state] = 1
                for s in self.states:
                    value_function[s] += alpha * (delta_t) * e_t[s]
                    e_t[s] = e_t[s] * lmbda
        return value_function
