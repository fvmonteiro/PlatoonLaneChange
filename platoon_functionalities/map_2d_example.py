from __future__ import annotations

from collections.abc import Iterable
import random
from typing import Callable

from platoon_functionalities import graph_explorer

import configuration

State = configuration.QuantizedState
Action2D = tuple[int, int]


def train(initial_state: State, problem_map: ProblemMap, max_episodes: int,
          epsilon: float = 0.5):
    # TODO: we're not dealing with self loop well, but whatever, right?

    sim_fun = problem_map.simulate
    initial_node = Node2D(initial_state, sim_fun)
    for episode in range(max_episodes):
        if (episode/max_episodes*100) % 10 == 0:
            print(f'Episode: {episode / max_episodes * 100:.2f}%')
            print(f'Best cost to go: {initial_node.best_cost_to_go}')
        # if episode / max_episodes >= 0.5:
        #     self._epsilon = self._default_epsilon

        current_node = initial_node
        path = []
        while not problem_map.is_goal_state(current_node.state):
            if current_node.is_new():
                current_node.generate_possible_actions()
            if random.uniform(0, 1) < epsilon:  # explore
                edge = current_node.explore()
            else:  # exploit
                edge = current_node.exploit()
            # Save
            path.append((current_node, edge))
            # Update
            current_node.update_best_edge(edge)
            # Advance
            current_node = edge.destination_node
        current_node.set_as_terminal()
        graph_explorer.update_along_path(path)
    print(initial_node.best_cost_to_go)
    
    
class Node2D(graph_explorer.Node):
    _possible_actions: list[Action2D]
    _explored_actions: dict[Action2D, Edge2D]
    _simulate: Callable[[State, Action2D], Node2D]

    def __init__(self, state: State,
                 sim_fun: Callable[[State, Action2D], 
                                   tuple[State, float]]):
        super().__init__(state, sim_fun)

    def generate_possible_actions(self):
        self._possible_actions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def set_as_terminal(self):
        self._best_cost_to_go = 0
        

class Edge2D(graph_explorer.Edge):
    _action: Action2D

    def __init__(self, destination_node: Node2D, action: Action2D, cost: float):
        super().__init__(destination_node, action, cost)

    @property
    def action(self) -> Action2D:
        return self._action
    
        
class ProblemMap:
    def __init__(self, grid_size: tuple[int, int],
                 obstacles: set[State], initial_state: State,
                 goal_states: Iterable[State]):
        self._grid_size = grid_size
        self._obstacles = obstacles
        if not self.is_state_free(initial_state):
            raise ValueError('Start node is not free')
        for gn in goal_states:
            if not self.is_state_free(gn):
                raise ValueError(f'Goal node {gn} is not free')
        self._start_node = initial_state
        self._goal_states = goal_states

    def is_state_free(self, state: State) -> bool:
        return (0 <= state[0] < self._grid_size[0]
                and 0 <= state[1] < self._grid_size[1]
                and state not in self._obstacles)

    def is_goal_state(self, state: State) -> bool:
        return state in self._goal_states

    def to_string(self):
        ret_str = ''
        for i in range(self._grid_size[0]):
            for j in range(self._grid_size[1]):
                state = (i, j)
                if state in self._obstacles:
                    ret_str += '|x|'
                elif state == self._start_node:
                    ret_str += '|o|'
                elif state in self._goal_states:
                    ret_str += '|$|'
                else:
                    ret_str += '| |'
            ret_str += '\n'
        return ret_str

    def simulate(self, state: State, action: Action2D
                 ) -> tuple[State, float]:
        new_state = (state[0] + action[0],
                     state[1] + action[1])
        if not self.is_state_free(new_state):
            cost = 1.e6
            # if new_node not in self._obstacles:  # allow traversing obstacles
            new_state = state
        else:
            cost = 1.
        return new_state, cost