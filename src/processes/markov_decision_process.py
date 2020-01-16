from abc import ABC, abstractmethod

from src.processes.markov_reward_process import MarkovRewardProcess


class MarkovDecisionProcess(MarkovRewardProcess, ABC):

    @abstractmethod
    def deo_something(self):
        pass
