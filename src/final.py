from typing import Sequence, Tuple, Mapping
from collections import defaultdict
import numpy as np

S = str
DataType = Sequence[Sequence[Tuple[S, float]]]
ProbFunc = Mapping[S, Mapping[S, float]]
RewardFunc = Mapping[S, float]
ValueFunc = Mapping[S, float]


def get_state_return_samples(
        data: DataType
) -> Sequence[Tuple[S, float]]:
    """
    prepare sequence of (state, return) pairs.
    Note: (state, return) pairs is not same as (state, reward) pairs.
    """
    return [(s, sum(r for (_, r) in l[i:]))
            for l in data for i, (s, _) in enumerate(l)]


def get_mc_value_function(
        state_return_samples: Sequence[Tuple[S, float]]
) -> ValueFunc:
    """
    Implement tabular MC Value Function compatible with the interface defined above.
    """
    rewards = defaultdict(float)
    counts = defaultdict(int)
    for state, r in state_return_samples:
        counts[state] += 1
        rewards[state] += r
    return {state: rewards[state]/counts[state] for state in rewards}


def get_state_reward_next_state_samples(
        data: DataType
) -> Sequence[Tuple[S, float, S]]:
    """
    prepare sequence of (state, reward, next_state) triples.
    """
    return [(s, r, l[i + 1][0] if i < len(l) - 1 else 'T')
            for l in data for i, (s, r) in enumerate(l)]


def get_probability_and_reward_functions(
        srs_samples: Sequence[Tuple[S, float, S]]
) -> Tuple[ProbFunc, RewardFunc]:
    """
    Implement code that produces the probability transitions and the
    reward function compatible with the interface defined above.
    """
    seqs_by_state = defaultdict(list)
    for sample in srs_samples:
        seqs_by_state[sample[0]].append(sample[1:])
    prob_func = defaultdict(dict)
    rewards = defaultdict(list)
    rewards['T'] = 0
    for state in seqs_by_state:
        counts = defaultdict(int)

        count = 0
        for sample in seqs_by_state[state]:
            counts[sample[1]] += 1
            rewards[state].append(sample[0])
            count += 1
        prob_func[state] = {s2: counts[s2] / count for s2 in counts}
    reward_func = {state: np.mean(rewards[state]) for state in rewards}
    return prob_func, reward_func


def get_mrp_value_function(
        prob_func: ProbFunc,
        reward_func: RewardFunc
) -> ValueFunc:
    """
    Implement code that calculates the MRP Value Function from the probability
    transitions and reward function, compatible with the interface defined above.
    Hint: Use the MRP Bellman Equation and simple linear algebra
    """

    value_func = defaultdict(int)

    states = reward_func.keys()
    transitions = np.zeros((len(states), len(states)))
    rewards = np.zeros(len(states))
    for i, s in enumerate(states):
        rewards[i] = reward_func[s]
        for j, s2 in enumerate(states):
            if s2 in prob_func[s]:
                transitions[i, j] = prob_func[s][s2]

    vf = np.linalg.inv(np.eye(len(states)) - transitions).dot(rewards)

    for i, s in enumerate(states):
        value_func[s] = vf[i]
    return value_func


def get_td_value_function(
        srs_samples: Sequence[Tuple[S, float, S]],
        num_updates: int = 300000,
        learning_rate: float = 0.3,
        learning_rate_decay: int = 30
) -> ValueFunc:
    """
    Implement tabular TD(0) (with experience replay) Value Function compatible
    with the interface defined above. Let the step size (alpha) be:
    learning_rate * (updates / learning_rate_decay + 1) ** -0.5
    so that Robbins-Monro condition is satisfied for the sequence of step sizes.
    """

    value_function = defaultdict(int)

    for i in range(num_updates):
        alpha = learning_rate * (i / learning_rate + 1) ** -0.5
        if i % len(srs_samples) == 0:
            np.random.shuffle(srs_samples)
        state, reward, next_state = srs_samples[i % len(srs_samples)]
        value_function[state] += alpha * (reward + value_function[next_state] - value_function[state])
    return value_function


def get_lstd_value_function(
        srs_samples: Sequence[Tuple[S, float, S]]
) -> ValueFunc:
    """
    Implement LSTD Value Function compatible with the interface defined above.
    Hint: Tabular is a special case of linear function approx where each feature
    is an indicator variables for a corresponding state and each parameter is
    the value function for the corresponding state.
    """
    states = list(filter(lambda x: x != 'T', set([s[0] for s in srs_samples] + [s[2] for s in srs_samples])))
    states_to_ind = {s: i for i, s in enumerate(states)}
    features = np.eye(len(states))

    sum_diffs = np.zeros((len(states), len(states)))
    sum_rewards = np.zeros(len(states))
    for s1, r1, s2 in srs_samples:
        s1_ind = states_to_ind[s1]
        sx1 = features[:, s1_ind].reshape((len(states), 1))
        if s2 != 'T':
            s2_ind = states_to_ind[s2]
            sx2 = features[:, s2_ind].reshape((len(states), 1))
        else:
            sx2 = np.zeros(sx1.shape)
        sum_diffs += sx1.dot((sx1 - sx2).T)
        sum_rewards += features[:, s1_ind] * r1

    w = np.linalg.inv(sum_diffs).dot(sum_rewards)
    value_func = defaultdict(int)
    for i, s in enumerate(states):
        value_func[s] = w[i]
    return value_func
