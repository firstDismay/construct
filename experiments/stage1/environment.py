from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import random
from collections import deque

Pos = Tuple[int, int]

@dataclass(frozen=True)
class Cost:
    time: int
    energy: int

@dataclass(frozen=True)
class EpisodeConfig:
    width: int = 7
    height: int = 7
    obstacle_probability: float = 0.12
    initial_time: int = 24
    initial_energy: int = 24

@dataclass
class Episode:
    seed: int
    grid: List[List[str]]
    start: Pos
    goal: Pos
    cost_law: Dict[str, Cost]

class Environment:
    """
    Stage I headless game environment.

    The agent observes local structure and objective resources, but the
    A/B movement-cost law is not exposed. The world is deterministic for
    a fixed seed, configuration and action sequence.
    """

    ACTIONS = ("MOVE_N", "MOVE_S", "MOVE_E", "MOVE_W", "SEARCH")

    def __init__(self, config: EpisodeConfig = EpisodeConfig()):
        self.config = config
        self.episode: Optional[Episode] = None
        self.position: Optional[Pos] = None
        self.time: int = 0
        self.energy: int = 0
        self.done: bool = False
        self.known_costs: Dict[str, Cost] = {}

    def generate(
        self,
        seed: int,
        swapped_law: bool = False,
        *,
        require_resource_feasible: bool = False,
    ) -> Episode:
        """Generate a deterministic world.

        With resource validation enabled, require a route that can be completed
        within the configured time and energy budgets under both the generated
        cost law and the A/B-swapped law. This is optional so tests can still
        construct deliberately resource-infeasible worlds.
        """
        rng = random.Random(seed)
        law = {
            "A": Cost(time=1, energy=4),
            "B": Cost(time=4, energy=1),
        }
        if swapped_law:
            law = {
                "A": Cost(time=4, energy=1),
                "B": Cost(time=1, energy=4),
            }
        alternate_law = {"A": law["B"], "B": law["A"]}

        for _ in range(1000):
            g = [
                [
                    "#" if rng.random() < self.config.obstacle_probability else rng.choice(("A", "B"))
                    for _ in range(self.config.width)
                ]
                for _ in range(self.config.height)
            ]
            free = [(x, y) for y in range(self.config.height)
                    for x in range(self.config.width) if g[y][x] != "#"]
            if len(free) < self.config.width * self.config.height // 2:
                continue

            start = rng.choice(free)
            goal = rng.choice(free)
            if start == goal or abs(start[0] - goal[0]) + abs(start[1] - goal[1]) < 6:
                continue

            g[start[1]][start[0]] = "S"
            g[goal[1]][goal[0]] = "G"

            ep = Episode(seed=seed, grid=g, start=start, goal=goal, cost_law=law)
            if not self._reachable(ep):
                continue
            if require_resource_feasible and not (
                self._resource_reachable(ep, law)
                and self._resource_reachable(ep, alternate_law)
            ):
                continue
            return ep

        raise RuntimeError(f"Could not generate episode for seed {seed}")

    def reset(self, episode: Episode, known_costs: Optional[Dict[str, Cost]] = None) -> dict:
        self.episode = episode
        self.position = episode.start
        self.time = self.config.initial_time
        self.energy = self.config.initial_energy
        self.done = False
        self.known_costs = dict(known_costs or {})
        return self.observe()

    def observe(self) -> dict:
        assert self.episode is not None and self.position is not None
        moves = []
        for action, delta in (
            ("MOVE_N", (0, -1)),
            ("MOVE_S", (0, 1)),
            ("MOVE_E", (1, 0)),
            ("MOVE_W", (-1, 0)),
        ):
            q = (self.position[0] + delta[0], self.position[1] + delta[1])
            if not self._in_bounds(q) or self.episode.grid[q[1]][q[0]] == "#":
                continue
            tile = self.episode.grid[q[1]][q[0]]
            moves.append({
                "action": action,
                "tile": tile,
                "cost_known": tile in self.known_costs,
                "cost": self.known_costs.get(tile),
            })

        return {
            "position": self.position,
            "goal": self.episode.goal,
            "time": self.time,
            "energy": self.energy,
            "moves": moves,
            "known_costs": dict(self.known_costs),
            "terminal": None if not self.done else ("WIN" if self.position == self.episode.goal else "LOSS"),
        }

    def step(self, action: str) -> tuple[dict, dict]:
        if self.done:
            raise RuntimeError("episode already finished")
        if action not in self.ACTIONS:
            raise ValueError(f"unknown action: {action}")

        assert self.episode is not None and self.position is not None
        before = self.observe()

        if action == "SEARCH":
            self.time -= 2
            revealed = {}
            for move in before["moves"]:
                tile = move["tile"]
                if tile in self.episode.cost_law:
                    self.known_costs[tile] = self.episode.cost_law[tile]
                    revealed[tile] = self.known_costs[tile]
            result = {"action": action, "revealed": revealed}
        else:
            delta = {
                "MOVE_N": (0, -1),
                "MOVE_S": (0, 1),
                "MOVE_E": (1, 0),
                "MOVE_W": (-1, 0),
            }[action]
            q = (self.position[0] + delta[0], self.position[1] + delta[1])
            if not self._in_bounds(q) or self.episode.grid[q[1]][q[0]] == "#":
                self.time -= 1
                result = {"action": action, "invalid": True}
            else:
                tile = self.episode.grid[q[1]][q[0]]
                cost = Cost(1, 1) if tile in ("S", "G") else self.episode.cost_law[tile]
                self.position = q
                self.time -= cost.time
                self.energy -= cost.energy
                result = {"action": action, "tile": tile, "cost": cost}

        if self.position == self.episode.goal:
            self.done = True
            terminal = "WIN"
        elif self.time <= 0 or self.energy <= 0:
            self.done = True
            terminal = "LOSS"
        else:
            terminal = None

        result["before"] = before
        result["after"] = self.observe()
        result["terminal"] = terminal
        return result, result["after"]

    def _in_bounds(self, p: Pos) -> bool:
        return 0 <= p[0] < self.config.width and 0 <= p[1] < self.config.height

    def _resource_reachable(self, ep: Episode, cost_law: Dict[str, Cost]) -> bool:
        """Check resource-feasible paths without exposing this to the agent."""
        initial = (ep.start, self.config.initial_time, self.config.initial_energy)
        queue = deque([initial])
        seen = {initial}
        deltas = ((0, -1), (0, 1), (1, 0), (-1, 0))

        while queue:
            position, time_left, energy_left = queue.popleft()
            x, y = position
            for dx, dy in deltas:
                target = (x + dx, y + dy)
                if not self._in_bounds(target) or ep.grid[target[1]][target[0]] == "#":
                    continue

                tile = ep.grid[target[1]][target[0]]
                cost = Cost(1, 1) if tile in ("S", "G") else cost_law[tile]
                next_time = time_left - cost.time
                next_energy = energy_left - cost.energy
                # step() checks the goal before resource exhaustion.
                if target == ep.goal:
                    return True
                if next_time <= 0 or next_energy <= 0:
                    continue

                state = (target, next_time, next_energy)
                if state not in seen:
                    seen.add(state)
                    queue.append(state)
        return False

    def _reachable(self, ep: Episode) -> bool:
        q = deque([ep.start])
        seen = {ep.start}
        while q:
            p = q.popleft()
            if p == ep.goal:
                return True
            x, y = p
            for dx, dy in ((0, -1), (0, 1), (1, 0), (-1, 0)):
                n = (x + dx, y + dy)
                if self._in_bounds(n) and ep.grid[n[1]][n[0]] != "#" and n not in seen:
                    seen.add(n)
                    q.append(n)
        return False
