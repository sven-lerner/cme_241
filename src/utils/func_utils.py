
def eq_to_epsilon(a: float, b: float, epsilon=1e-8) -> bool:
    return abs(a - b) <= epsilon
