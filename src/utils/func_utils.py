
epsilon = 1e-8


def eq_to_epsilon(a: float, b: float) -> bool:
    return abs(a - b) <= epsilon
