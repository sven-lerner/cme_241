from abc import ABC, abstractmethod
from collections import defaultdict

from src.utils.generic_typevars import S, A
from typing import Mapping, Set, Tuple, Generic, Sequence

from src.processes.markov_reward_process import MarkovRewardProcess, BaseMarkovRewardProcessImpl
from src.utils.typevars import MDPTransitions, MDPRewards, MDPActions


class MarkovDecisionProcess(MarkovRewardProcess, ABC):

    @abstractmethod
    def get_actions_by_states(self):
        pass

    @abstractmethod
    def get_state_action_transitions(self):
        pass

    @abstractmethod
    def get_mrp(self, policy: Mapping[S, Set[Tuple[A, float]]]) -> MarkovRewardProcess:
        pass

    @abstractmethod
    def get_rewards(self) -> MDPRewards:
        pass

    @abstractmethod
    def get_discount(self) -> float:
        pass


class BaseMarkovDecisionProcessImpl(MarkovDecisionProcess, BaseMarkovRewardProcessImpl):

    def get_discount(self) -> float:
        return self._gamma

    def get_rewards(self):
        return self._rewards

    def get_state_action_transitions(self):
        return self._transitions

    def get_terminal_states(self) -> Sequence[S]:
        return self._terminal_states

    def get_actions_by_states(self):
        return self._actions_by_state

    def __init__(self, transitions: MDPTransitions,
                 rewards: MDPRewards,
                 terminal_states: Set[S],
                 gamma: float):
        self._states = sorted(transitions.keys())
        actions_by_state: MDPActions = {s: list(transitions[s].keys()) for s in
                                        self._states}
        self._rewards = rewards
        self._actions_by_state = actions_by_state
        self._terminal_states = terminal_states
        self._non_terminal_sates = self.get_non_terminal_states()
        self._transitions = transitions
        self._gamma = gamma

    def get_mrp(self, policy: Mapping[S, Set[Tuple[A, float]]]):
        mrp_transitions = {s: defaultdict(int) for s in self._states}
        for s in self._states:
            for action, action_prob in policy[s]:
                for next_state, next_state_prob in self._transitions[s][action].items():
                    mrp_transitions[s][next_state] += action_prob*next_state_prob
        mrp_rewards: Mapping[S, float] = {s: sum(
            [self._rewards[s][a]*action_prob for a, action_prob in policy[s]]
            ) for s in self._states}
        print('mrp transitions', mrp_transitions)
        print('mrp rewards', mrp_rewards)
        return BaseMarkovRewardProcessImpl(mrp_transitions, mrp_rewards, self._gamma)
