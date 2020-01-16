from typing import Callable, Sequence, Mapping, Tuple
from src.utils.generic_typevars import S, A

SSf = Mapping[S, Mapping[S, float]]
SSTff = Mapping[S, Mapping[S, Tuple[float, float]]]
STSff = Mapping[S, Tuple[Mapping[S, float], float]],
SAf = Mapping[S, Mapping[A, float]]
SASf = Mapping[S, Mapping[A, Mapping[S, float]]]
SASTff = Mapping[S, Mapping[A, Mapping[S, Tuple[float, float]]]]
SATSff = Mapping[S, Mapping[A, Tuple[Mapping[S, float], float]]]
