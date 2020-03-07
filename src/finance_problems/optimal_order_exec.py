import numpy as np

from src.control.iterative_methods import policy_iteration
from src.processes.markov_decision_process import BaseMarkovDecisionProcessImpl
from src.rl_algos.control import QLearningTabular


class Market:

    def __init__(self, initial_price):
        self.init_price = initial_price
        self.history = []
        self.price = initial_price
        self.orders_placed = []

    def get_price(self):
        return self.price

    def reset(self):
        self.price = self.init_price
        self.history = []
        self.orders_placed = []

    def process_timestep(self):
        pass

    def submit_order(self, num_shares):
        pass


class LPIMarket(Market):

    def __init__(self, initial_price, beta, alpha):
        super().__init__(initial_price)
        self.beta = beta
        self.alpha = alpha

    def process_timestep(self):
        shares_sold = np.sum(self.orders_placed)
        self.history.append((self.price, shares_sold))
        self.orders_placed = []
        self.price = self.price - self.alpha * shares_sold + (np.random.rand() - 0.5)

    def submit_order(self, num_shares):
        self.orders_placed.append(num_shares)
        return self.price - self.beta * num_shares


class DumpStockRLProblem():

    def __init__(self, T: int, num_shares: int, market: Market):
        self.T = T
        self.market = market
        self.init_shares = num_shares
        self.states = [(T, Nt) for T in range(1, self.T + 1) for Nt in range(0, num_shares + 1)] + [(0, 0)] + [(-1, 0)]
        self.state_actions = self.get_state_actions()
        self.transitions = self.get_transitions()

    def get_state_actions(self):
        return dict((state, list(range(state[1] + 1))) if state[0] > 1 else (state, [state[1]])
                    for state in self.states)

    def get_submit_order_process_timestep_callable(self, num_shares, T, rt):
        def call():
            price_sold = self.market.submit_order(num_shares)
            self.market.process_timestep()
            return (T - 1, rt - num_shares), price_sold * num_shares

        return call

    def get_transitions(self):
        transitions = {}
        for state in self.states:
            for action in self.state_actions[state]:
                transitions[(state, action)] = self.get_submit_order_process_timestep_callable(action, state[0],
                                                                                               state[1])
        return transitions

    def solve(self, max_episodes):
        terminal_states = [state for state in self.states if state[0] == 0]
        starting_state_distributions = [((start_time, start_shares), 1 / (self.T * self.init_shares))
                                        for start_time in range(1, self.T + 1) for start_shares in
                                        range(1, self.init_shares + 1)]

        dump_stock_ql = QLearningTabular(self.transitions, terminal_states=terminal_states,
                                         state_actions=self.state_actions, gamma=1, num_episodes=int(max_episodes), max_iter=100,
                                         starting_distribution=starting_state_distributions)
        q_func = dump_stock_ql.learn_q_value_function(alpha=1e-2, reset=lambda: self.market.reset())
        policy = dump_stock_ql.get_epsilon_greedy_policy(q_func, 0)
        return policy, q_func


class DumpStockMDP(BaseMarkovDecisionProcessImpl):

    def __init__(self, T: int, num_shares: int, sigma2: float, alpha: float, beta: float,
                 min_price: float = 0, max_price: float = 60):
        # sigma2 is the var of the epsilon term in distribution of returns of stock
        self.T = T
        self.init_shares = num_shares
        self.alpha = alpha
        self.beta = beta
        self.max_price = max_price
        self.min_price = min_price
        # assume prices between 0 and 100, quantize to every $1
        self.states = [(T, Pt, Nt) for T in range(1, self.T + 1)
                       for Pt in range(min_price, max_price + 1) for Nt in range(0, num_shares + 1)] + \
                      [(0, Pt, 0) for Pt in range(min_price, max_price + 1)]
        self.state_actions = self.get_state_actions()
        self.mdp_transitions, self.mdp_rewards = self.get_transitions_and_rewards(uncertain_price=True)
        self.terminal_states = [state for state in self.states if state[0] == 0]
        super().__init__(transitions=self.mdp_transitions, rewards=self.mdp_rewards,
                         terminal_states=self.terminal_states,
                         gamma=1)

    def get_state_actions(self):
        return dict((state, list(range(state[2] + 1))) if state[0] > 1 else (state, [state[2]])
                    for state in self.states)

    def get_transitions_and_rewards(self, uncertain_price):
        transitions = {state: {} for state in self.states}
        rewards = {state: {} for state in self.states}
        for state in self.states:
            if state[0] > 0:
                assert len(self.state_actions[state]) > 0
                for action in self.state_actions[state]:
                    sale_price = state[1] - self.beta * action
                    next_price = max(int(state[1] - self.beta * action), 0)
                    next_state = (state[0] - 1, next_price, state[2] - action)
                    reward = sale_price * action
                    if uncertain_price:
                        if next_price == self.min_price:
                            transitions[state][action] = {
                                (state[0] - 1, next_price, state[2] - action): 3 / 4,
                                (state[0] - 1, min(next_price + 1, self.max_price), state[2] - action): 1 / 4,
                            }
                        elif next_price == self.max_price:
                            transitions[state][action] = {
                                (state[0] - 1, max(next_price - 1, self.min_price), state[2] - action): 1 / 4,
                                (state[0] - 1, next_price, state[2] - action): 3 / 4
                                }
                        else:
                            transitions[state][action] = {
                                (state[0] - 1, max(next_price - 1, self.min_price), state[2] - action): 1 / 4,
                                (state[0] - 1, next_price, state[2] - action): 1 / 2,
                                (state[0] - 1, min(next_price + 1, self.max_price), state[2] - action): 1 / 4,
                                }

                    else:
                        transitions[state][action] = {(state[0] - 1, next_price, state[2] - action): 1}
                    rewards[state][action] = reward
            else:
                transitions[state][0] = {state: 1}
                rewards[state][0] = 0
        return transitions, rewards

    def solve(self):
        policy = policy_iteration(self)
        return policy
