from experiments.stage1.environment import Environment, EpisodeConfig

def test_reproducible_generation():
    env = Environment()
    a = env.generate(1234)
    b = env.generate(1234)
    assert a.grid == b.grid
    assert a.start == b.start
    assert a.goal == b.goal
    assert a.cost_law == b.cost_law

def test_hidden_cost_law():
    env = Environment()
    ep = env.generate(1234)
    obs = env.reset(ep)
    assert obs["known_costs"] == {}
    assert all(not x["cost_known"] for x in obs["moves"])

def test_search_reveals_only_observed_types():
    env = Environment()
    ep = env.generate(1234)
    obs = env.reset(ep)
    result, _ = env.step("SEARCH")
    assert set(result["revealed"]).issubset(set(ep.cost_law))

def test_resources_are_objective():
    env = Environment(EpisodeConfig(initial_time=2, initial_energy=24))
    ep = env.generate(1234)
    env.reset(ep)
    result, obs = env.step("SEARCH")
    assert obs["time"] == 0
    assert result["terminal"] == "LOSS"

def test_goal_is_binary_terminal():
    env = Environment()
    ep = env.generate(1234)
    env.reset(ep)
    assert env.observe()["terminal"] is None
    assert env.observe()["position"] != ep.goal
