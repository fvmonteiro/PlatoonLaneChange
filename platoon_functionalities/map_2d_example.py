from __future__ import annotations

from collections.abc import Iterable
from typing import Callable
# import weakref

from platoon_functionalities import graph_explorer

import configuration
from platoon_functionalities.graph_explorer import Node, ActionBase

State = configuration.QuantizedState
Action2D = tuple[int, int]


def train(initial_state: State, max_episodes: int, problem_map: ProblemMap,
          epsilon: float = 0.5, verbose_level: int = 0):
    # TODO: we're not dealing with self loop well, but whatever, right?

    sim_fun = problem_map.simulate

    def is_goal_node(node: Node2D) -> bool:
        if problem_map.is_goal_state(node.state):
            node.set_as_terminal()
            return True
        return False

    initial_node = Node2D(initial_state, sim_fun)
    graph_explorer.train_base(initial_node, max_episodes,
                              is_goal_node, epsilon, verbose_level)


class Node2D(graph_explorer.Node):
    _possible_actions: list[Action2D]
    _explored_actions: dict[Action2D, Edge2D]
    _simulate: Callable[[State, Action2D], tuple[State, float]]

    def __init__(
            self, state: State,
            sim_fun: Callable[[State, Action2D], tuple[State, float]]
    ):
        super().__init__(state, sim_fun)

    def generate_possible_actions(self):
        self._possible_actions = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def add_edge_from_state(self, successor_state: State, action: Action2D,
                            cost: float) -> None:
        if successor_state == self.state:
            next_node = self
        else:
            next_node = Node2D(successor_state, self._simulate)
        self.add_edge_from_node(next_node, action, cost)
        # self._explored_actions[action] = Edge2D(next_node, action, cost)

    def add_edge_from_node(self, successor: Node2D, action: Action2D,
                           cost: float) -> None:
        if self == successor:
            self._possible_actions.remove(action)
        self._explored_actions[action] = Edge2D(successor, action, cost)


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
