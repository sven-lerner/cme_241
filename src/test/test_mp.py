from src.processes.markov_process import BaseMarkovProcessImpl


def test_get_stationary_empty():
    mp = BaseMarkovProcessImpl({})
    stationary = mp.get_stationary_distributions()
    assert len(stationary) == 0


def test_get_stationary_simple():
    toy_mdp = {1: {1: 0, 2: 1/3, 3: 1/3, 4: 1/3},
               2: {1: 1/3, 2: 0, 3: 1/3, 4: 1/3},
               3: {1: 1/3, 2: 1/3, 3: 0, 4: 1/3},
               4: {1: 1/3, 2: 1/3, 3: 1/3, 4: 0},
               }
    mp = BaseMarkovProcessImpl(toy_mdp)
    stationary = mp.get_stationary_distributions()
    print(stationary)
    assert stationary == [{1: 1/4, 2: 1/4, 3: 1/4, 4: 1/4}]


def test_get_stationary_sink():
    toy_mdp = {1: {1: 1},
               2: {1: 1 / 3, 2: 0, 3: 1 / 3, 4: 1 / 3},
               3: {1: 1 / 3, 2: 1 / 3, 3: 0, 4: 1 / 3},
               4: {1: 1 / 3, 2: 1 / 3, 3: 1 / 3, 4: 0},
               }
    mp = BaseMarkovProcessImpl(toy_mdp)
    stationary = mp.get_stationary_distributions()
    print(stationary)
    assert stationary == [{1: 1, 2: 0, 3: 0, 4: 0}]
