from abc import ABC, abstractmethod
from typing import Mapping, Sequence
import numpy as np
from src.processes.markov_process import MarkovProcess, BaseMarkovProcessImpl
from src.utils.func_utils import eq_to_epsilon
from src.utils.generic_typevars import S
from src.utils.typevars import SSf


class MarkovRewardProcess(MarkovProcess, ABC):

	@abstractmethod
	def get_value_func_vec(self) -> np.ndarray:
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

	def __init__(self, state_transitions: SSf, rewards: Mapping[S, float], gamma: float):
		super().__init__(state_transitions)
		self._rewards = rewards
		self._gamma = gamma
		self._terminal_states = self.get_terminal_states()
		self._non_terminal_states = self.get_non_terminal_states()

	def get_non_terminal_states(self) -> Sequence[S]:
		return sorted([s for s in self._states if s not in self._terminal_states])

	def get_transitions_matrix(self):
		transition_matrix = np.zeros((len(self._states), len(self._states)))
		for i, s in enumerate(self._states):
			for j, s_prime in enumerate(self._states):
				transition_matrix[i][j] = self._state_transitions[s].get(s_prime, 0)
		return transition_matrix.T

	def get_terminal_states(self) -> Sequence[S]:
		return sorted([s for s in self._states if eq_to_epsilon(self._state_transitions[s].get(s, 0), 1)])

	def get_reward_vector(self) -> np.ndarray:
		rewards = np.zeros((len(self._states), 1))
		for i, s in enumerate(self._states):
			rewards[i] = self._rewards[s]
		return rewards

	def get_value_func_vec(self) -> np.ndarray:
		rewards = self.get_reward_vector()
		transitions = self.get_transitions_matrix()
		values = np.linalg.inv(np.eye(transitions.shape[0]) - self._gamma*transitions).dot(rewards)
		return values
