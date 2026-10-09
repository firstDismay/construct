from experiments.stage1.environment import Environment, EpisodeConfig


def swapped_law(law):
    return {"A": law["B"], "B": law["A"]}


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
    assert all(not move["cost_known"] for move in obs["moves"])


def test_search_reveals_only_adjacent_observed_types():
    env = Environment()
    ep = env.generate(1234)
    obs = env.reset(ep)
    expected_tiles = {
        move["tile"] for move in obs["moves"] if move["tile"] in ep.cost_law
    }
    result, _ = env.step("SEARCH")
    assert set(result["revealed"]) == expected_tiles
    for tile, cost in result["revealed"].items():
        assert cost == ep.cost_law[tile]


def test_resources_are_objective():
    env = Environment(EpisodeConfig(initial_time=2, initial_energy=24))
    ep = env.generate(1234)
    env.reset(ep)
    result, obs = env.step("SEARCH")
    assert obs["time"] == 0
    assert result["terminal"] == "LOSS"


def test_goal_transition_wins_even_when_final_move_exhausts_resources():
    env = Environment()
    ep = env.generate(1234)
    env.reset(ep)
    gx, gy = ep.goal
    candidates = [
        ((gx, gy - 1), "MOVE_S"),
        ((gx, gy + 1), "MOVE_N"),
        ((gx - 1, gy), "MOVE_E"),
        ((gx + 1, gy), "MOVE_W"),
    ]
    for neighbor, action in candidates:
        x, y = neighbor
        if env._in_bounds(neighbor) and ep.grid[y][x] != "#":
            env.position = neighbor
            env.time = 1
            env.energy = 1
            result, obs = env.step(action)
            assert obs["position"] == ep.goal
            assert result["terminal"] == "WIN"
            assert obs["terminal"] == "WIN"
            return
    raise AssertionError("Generated goal had no traversable neighbor")


def test_generation_can_require_resource_feasibility_under_both_laws():
    env = Environment()
    for seed in range(20):
        ep = env.generate(seed, require_resource_feasible=True)
        assert env._resource_reachable(ep, ep.cost_law)
        assert env._resource_reachable(ep, swapped_law(ep.cost_law))


def test_swapping_law_does_not_change_generated_geometry():
    env = Environment()
    normal = env.generate(1234, require_resource_feasible=True)
    swapped = env.generate(1234, swapped_law=True, require_resource_feasible=True)
    assert normal.grid == swapped.grid
    assert normal.start == swapped.start
    assert normal.goal == swapped.goal
    assert normal.cost_law == swapped_law(swapped.cost_law)
