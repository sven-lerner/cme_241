from src.finance_problems.optimal_order_exec import LPIMarket, DumpStockRLProblem, DumpStockMDP


def test_simple_order_exec_rl():
    market = LPIMarket(25, alpha=0.01, beta=1)
    dsp = DumpStockRLProblem(2, 8, market)
    policy, q_func = dsp.solve(max_episodes=1e4)
    assert policy.get_action((2, 4)) == 2
    assert policy.get_action((2, 8)) == 4


def test_simple_order_exec_mdp():
    dsmdp = DumpStockMDP(2, 8, 1, 0.1, 1, min_price=0,max_price=10)
    opt_policy = dsmdp.solve()
    assert opt_policy[(2, 5, 4)] == {(2, 1)}
    assert opt_policy[(2, 5, 8)] == {(4, 1)}
