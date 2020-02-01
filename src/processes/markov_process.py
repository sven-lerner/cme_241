from abc import ABC, abstractmethod
from src.utils.typevars import MPTransitions
from src.utils.generic_typevars import S
from typing import Mapping, Set, Generic, Sequence
import numpy as np


class MarkovProcess(ABC, Generic[S]):

    @abstractmethod
    def get_transition_probabilities(self) -> Mapping[S, Mapping[S, float]]:
        pass

    @abstractmethod
    def get_stationary_distributions(self) -> Mapping[S, float]:
        pass


class BaseMarkovProcessImpl(MarkovProcess):

    def __init__(self, state_transitions: MPTransitions):
        self._state_transitions = state_transitions
        self._states = sorted(state_transitions.keys())

    def get_transitions_matrix(self):
        transition_matrix = np.zeros((len(self._states), len(self._states)))
        for i, s in enumerate(self._states):
            for j, s_prime in enumerate(self._states):
                transition_matrix[i][j] = self._state_transitions[s].get(s_prime, 0)
        return transition_matrix.T

    def get_transition_probabilities(self) -> Mapping[S, Mapping[S, float]]:
        return self._state_transitions

    def get_stationary_distributions(self) -> Sequence[Mapping[S, float]]:
        eig_vals, eig_rvects = np.linalg.eig(self.get_transitions_matrix())
        one_eig_vects = eig_rvects[:, np.abs(eig_vals - 1) < 1e-8].T
        retlist = []
        for eig_vect in one_eig_vects:
            eig_vect = eig_vect / np.sum(eig_vect)
            retlist.append({s: eig_vect[i] for i, s in enumerate(self._states)})
        return retlist
