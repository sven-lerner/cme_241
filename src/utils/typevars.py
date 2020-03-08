from typing import Callable, Mapping, Tuple
from src.utils.generic_typevars import S, A

# Markov Process Types
MPTransitions = Mapping[S, Mapping[S, float]]

# Markov Reward Process Types
MRPRewards = Mapping[S, float]

# Markov Decision Process Types
MDPTransitions = Mapping[S, Mapping[A, Mapping[S, float]]]
MDPActions = Mapping[S, A]
MDPRewards = Mapping[S, Mapping[A, float]]
Tab_RL_Transitions = Mapping[Tuple[S, A], Callable[[], Tuple[S, float]]]
