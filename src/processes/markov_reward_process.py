from abc import ABC, abstractmethod
from typing import Mapping, Sequence
import numpy as np
from src.processes.markov_process import MarkovProcess, BaseMarkovProcessImpl
from src.utils.func_utils import eq_to_epsilon
from src.utils.generic_typevars import S
from src.utils.typevars import MPTransitions, MRPRewards


class MarkovRewardProcess(MarkovProcess, ABC):

    @abstractmethod
    def get_value_func_vec(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_value_func(self) -> Mapping[S, float]:
        pass

    @abstractmethod
    def get_reward_vector(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_terminal_states(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_non_terminal_states(self) -> np.ndarray:
        pass


class BaseMarkovRewardProcessImpl(MarkovRewardProcess, BaseMarkovProcessImpl):

    def __init__(self, state_transitions: MPTransitions, rewards: MRPRewards, gamma: float):
        super().__init__(state_transitions)
        self._rewards = rewards
        self._gamma = gamma
        self._terminal_states = self.get_terminal_states()
        self._non_terminal_states = self.get_non_terminal_states()

    def get_non_terminal_states(self) -> Sequence[S]:
        return sorted([s for s in self._states if s not in self._terminal_states])

    def get_transitions_matrix(self):
        dim = len(self._non_terminal_states)
        transition_matrix = np.zeros((dim, dim))
        for i, s in enumerate(self._non_terminal_states):
            for j, s_prime in enumerate(self._non_terminal_states):
                transition_matrix[i][j] = self._state_transitions[s].get(s_prime, 0)
        return transition_matrix

    def get_terminal_states(self) -> Sequence[S]:
        return sorted([s for s in self._states if eq_to_epsilon(self._state_transitions[s].get(s, 0), 1)])

    def get_reward_vector(self) -> np.ndarray:
        rewards = np.zeros((len(self._non_terminal_states), 1))
        for i, s in enumerate(self._non_terminal_states):
            rewards[i] = self._rewards[s]
        return rewards

    def get_value_func_vec(self) -> np.ndarray:
        rewards = self.get_reward_vector()
        transitions = self.get_transitions_matrix()
        values = np.linalg.inv(np.eye(transitions.shape[0]) - self._gamma * transitions).dot(rewards)
        return values

    def get_value_func(self) -> Mapping[S, float]:
        value_func_vect = self.get_value_func_vec()
        value_func = {s: value_func_vect[i][0] for i, s in enumerate(self._non_terminal_states)}
        for s in self._terminal_states:
            value_func[s] = self._rewards[s]
        return value_func
