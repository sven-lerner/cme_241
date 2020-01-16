from abc import ABC, abstractmethod

from src.processes.markov_process import MarkovProcess


class MarkovRewardProcess(MarkovProcess, ABC):

	@abstractmethod
	def get_rewards(self):
		pass
