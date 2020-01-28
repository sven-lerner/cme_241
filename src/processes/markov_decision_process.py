from abc import ABC, abstractmethod
from collections import defaultdict

from src.utils.generic_typevars import S, A
from typing import Mapping, Set, Tuple, Generic

from src.processes.markov_reward_process import MarkovRewardProcess, BaseMarkovRewardProcessImpl


class MarkovDecisionProcess(MarkovRewardProcess, ABC):

    @abstractmethod
    def get_mrp(self, policy):
        pass


class BaseMarkovDecisionProcessImpl(MarkovDecisionProcess, BaseMarkovRewardProcessImpl):

    def __init__(self, transitions: Mapping[S, Mapping[A, Mapping[S, float]]], rewards: Mapping[S, Mapping[A, float]],
                 actions_by_state: Mapping[S, Set[A]], terminal_states: Set[S], gamma: float):
        super().__init__()
        self._states = sorted(transitions.keys())
        self._rewards = rewards
        self._actions_by_state = actions_by_state
        self._terminal_states = terminal_states
        self._non_terminal_sates = self.get_non_terminal_states()
        self._transitions = transitions
        self._gamma = gamma

    def get_mrp(self, policy: Mapping[S, Set[(A, float)]]):
        mrp_transitions = {s: defaultdict(int) for s in self._states}
        for s in self._states:
            for action, action_prob in policy[s]:
                for next_state, next_state_prob in self._transitions[s][action].items():
                    mrp_transitions[s][next_state] += action_prob*next_state_prob
        mrp_rewards: Mapping[S, float] = {s: sum(
            [self._rewards[s][a]*action_prob for a, action_prob in policy[s]]
            ) for s in self._states}
        return BaseMarkovRewardProcessImpl(mrp_transitions, mrp_rewards, self._gamma)
