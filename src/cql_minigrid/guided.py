"""Separate shortest-path demonstrations; the deployed policy remains a neural CQL policy."""
from collections import deque
from pathlib import Path

import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX

from .collector import collect_dataset
from .common import resource_path, sha256
from .rollout import run_episodes

TEACHER_VERSION = 'symbolic-pose-bfs-replan-v1'


def shortest_path_policy(obs, rng):
    """Replan over position/direction after every action as obstacles move."""
    n_objects = len(OBJECT_TO_IDX)
    grid_size = 8 * 8 * (n_objects + 2)
    objects = obs[:grid_size].reshape(8, 8, n_objects + 2)[..., :n_objects].argmax(-1)
    agent = np.argwhere(objects == OBJECT_TO_IDX['agent'])
    goal = np.argwhere(objects == OBJECT_TO_IDX['goal'])
    if len(agent) != 1 or len(goal) != 1:
        raise ValueError('The teacher requires one visible agent and goal')
    x, y = map(int, agent[0])
    destination = tuple(map(int, goal[0]))
    direction = int(obs[grid_size:grid_size + 4].argmax())
    queue = deque([((x, y, direction), None)])
    seen = {(x, y, direction)}
    vectors = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    while queue:
        (x, y, d), first = queue.popleft()
        if (x, y) == destination:
            return first
        dx, dy = vectors[d]
        for action, state in [(2, (x + dx, y + dy, d)),
                              (0, (x, y, (d - 1) % 4)), (1, (x, y, (d + 1) % 4))]:
            nx, ny, _ = state
            if not (0 <= nx < 8 and 0 <= ny < 8):
                continue
            if action == 2 and objects[nx, ny] not in [OBJECT_TO_IDX['empty'], OBJECT_TO_IDX['goal']]:
                continue
            if state not in seen:
                seen.add(state)
                queue.append((state, action if first is None else first))
    # Obstacles can temporarily block all paths; turn and observe their next positions.
    return 0


def collect_guided(config, root, *, config_path=None):
    import json
    settings = config['guided']
    seeds = json.loads(resource_path(config['seeds_file'], config_path).read_text())
    audit = {'teacher_version': TEACHER_VERSION,
             **run_episodes(shortest_path_policy, seeds['validation'])}
    return collect_dataset(Path(root) / 'datasets/guided.npz', {'planner': shortest_path_policy},
                           {'planner': 1.}, target=settings['transitions'],
                           episode_seed_start=settings['episode_seed_start'],
                           policy_seed_start=settings['policy_seed_start'],
                           source_hashes={'planner': sha256(__file__)}, source_audit=audit)
