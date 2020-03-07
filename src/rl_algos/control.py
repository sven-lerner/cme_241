import numpy as np
from src.control.iterative_methods import policy_iteration
from src.processes.markov_decision_process import BaseMarkovDecisionProcessImpl
from typing import Callable, Sequence, Mapping, Tuple, Set

from src.rl_algos.base_rl_algo import BaseTabularRL, Policy
from src.utils.generic_typevars import S, A
from src.utils.typevars import Tab_RL_Transitions
from tqdm.notebook import tqdm
from collections import defaultdict



class SarsaRL(BaseTabularRL):

    def run_episode_from_q(self, starting_state: S, q_val_function: Mapping[Tuple[S ,A], float], epsilon,
                           alpha) -> Sequence[Tuple[S, A, float, bool]]:
        curr_state = starting_state
        iteration = 0
        continue_iter = True

        while continue_iter:
            iteration += 1

            policy = self.get_epsilon_greedy_policy(q_val_function, epsilon)
            action = policy.get_action(curr_state)
            next_state, reward = self.transitions[(curr_state, action)]()
            next_action = policy.get_action(next_state)
            q_val_function[(curr_state, action)] += alpha *(reward + \
                                                             self.gamma * q_val_function[(next_state, next_action)] - \
                                                             q_val_function[(curr_state, action)])

            if iteration >= self.max_iter or curr_state in self.terminal_states:
                continue_iter = False
            curr_state = next_state
        return q_val_function

    def learn_q_value_function(self, alpha, reset=None) -> Mapping[Tuple[S, A], float]:
        current_q_vf = {(state, a): 0 for state in self.states for a in self.state_actions[state]}
        for k in tqdm(range(1, self.num_episodes)):
            if reset is not None:
                reset()
            epsilon = 1 / k
            starting_state_index = np.random.choice([i for i, _ in enumerate(self.starting_distribution)], 1,
                                                    p=[prob for _, prob in self.starting_distribution])[0]
            starting_state = self.starting_distribution[starting_state_index][0]
            current_q_vf = self.run_episode_from_q(starting_state, current_q_vf, epsilon, alpha)
        return current_q_vf


class SarsaLambdaRL(BaseTabularRL):

    def run_episode_from_q(self, starting_state: S, q_val_function: Mapping[Tuple[S, A], float], epsilon,
                           alpha, lmbda) -> Sequence[Tuple[S, A, float, bool]]:
        visited = set()
        curr_state = starting_state
        iteration = 0
        continue_iter = True
        episode = []
        e_t = defaultdict(int)
        while continue_iter:
            iteration += 1
            first_visit = curr_state not in visited
            policy = self.get_epsilon_greedy_policy(q_val_function, epsilon)
            action = policy.get_action(curr_state)
            next_state, reward = self.transitions[(curr_state, action)]()
            next_action = policy.get_action(next_state)
            delta = reward + self.gamma * q_val_function[(next_state, next_action)] - q_val_function[
                (curr_state, action)]
            e_t[(curr_state, action)] = 1
            for s in self.states:
                for a in self.state_actions[s]:
                    q_val_function[(s, a)] += alpha * e_t[(s, a)] * delta
                    e_t[(s, a)] = e_t[(s, a)] * lmbda
            if iteration >= self.max_iter or curr_state in self.terminal_states:
                continue_iter = False
            curr_state = next_state
        return q_val_function

    def learn_q_value_function(self, alpha, lmbda) -> Mapping[Tuple[S, A], float]:
        current_q_vf = {(state, a): 0 for state in self.states for a in self.state_actions[state]}
        for k in range(1, self.num_episodes):
            epsilon = 1 / k
            starting_state = np.random.choice([state for state, _ in self.starting_distribution], 1,
                                              p=[prob for _, prob in self.starting_distribution])[0]
            current_q_vf = self.run_episode_from_q(starting_state, current_q_vf, epsilon, alpha, lmbda)
        return current_q_vf


class QLearningTabular(BaseTabularRL):

    def run_episode_update_q(self, starting_state: S, q_val_function: Mapping[Tuple[S, A], float], epsilon,
                             alpha, counts_per_state, base_policy: Policy = None) -> Sequence[Tuple[S, A, float, bool]]:
        visited = set()
        curr_state = starting_state
        iteration = 0
        continue_iter = True
        greedy_policy_for_episode = self.get_epsilon_greedy_policy(q_val_function, 0)
        while continue_iter:
            iteration += 1
            first_visit = curr_state not in visited
            if base_policy is None:
                action = self.get_epsilon_greedy_policy(q_val_function, epsilon).get_action(curr_state)
            else:
                action = base_policy.get_action(curr_state)

            next_state, reward = self.transitions[(curr_state, action)]()

            greedy_next_action = greedy_policy_for_episode.get_action(next_state)
            counts_per_state[(curr_state, action)] += 1
            weight = max(1 / counts_per_state[(curr_state, action)], 1e-3)
            q_val_function[(curr_state, action)] += weight * (
                        reward + self.gamma * q_val_function[(next_state, greedy_next_action)] \
                        - q_val_function[(curr_state, action)])

            if iteration >= self.max_iter or curr_state in self.terminal_states:
                continue_iter = False
            curr_state = next_state
        return q_val_function

    def learn_q_value_function(self, alpha, base_policy=None, reset=None) -> Mapping[Tuple[S, A], float]:
        current_q_vf = {(state, a): 0 for state in self.states for a in self.state_actions[state]}
        counts_per_state = defaultdict(int)
        for k in tqdm(range(1, self.num_episodes)):
            if reset is not None:
                reset()
            epsilon = 1 / (max(1, k / 1000))
            base_policy = self.get_epsilon_greedy_policy(current_q_vf, epsilon)
            starting_state_index = np.random.choice([i for i, _ in enumerate(self.starting_distribution)], 1,
                                                    p=[prob for _, prob in self.starting_distribution])[0]
            starting_state = self.starting_distribution[starting_state_index][0]
            current_q_vf = self.run_episode_update_q(starting_state, current_q_vf, epsilon, alpha,
                                                     counts_per_state,
                                                     base_policy)
        return current_q_vf
