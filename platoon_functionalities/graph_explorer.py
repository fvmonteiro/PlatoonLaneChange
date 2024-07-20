from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from functools import partial
import random
from typing import Any, Callable

import numpy as np

import configuration

ActionBase = Any  # tuple[int, int]
State = configuration.QuantizedState


def update_along_path(path: list[tuple[Node, Edge]]):
    for node, edge in path[::-1]:
        node.update_best_edge(edge)


class Node (ABC):

    _state: State
    _possible_actions: list[ActionBase]
    _explored_actions: dict[ActionBase, Edge]
    _best_edge: Edge
    _best_cost_to_go: float
    _simulate: Callable[[State, ActionBase], tuple[State, float]]

    def __init__(self, state: State,
                 sim_fun: Callable[[State, ActionBase], tuple[State, float]]):
        self._state = state
        self._possible_actions = []
        self._explored_actions = dict()  # needed to prevent re-simulation
        self._best_cost_to_go = np.inf
        self._simulate = sim_fun

    # create getters for all members
    @property
    def state(self) -> State:
        return self._state

    @property
    def best_cost_to_go(self) -> float:
        return self._best_cost_to_go

    def is_new(self):
        return len(self._possible_actions) == 0

    @abstractmethod
    def generate_possible_actions(self):
        pass

    def exploit(self) -> Edge:
        if len(self._explored_actions) == 0:
            return self.explore()
        # try:
        return self._best_edge
        # except AttributeError:
        #     return self._explored_actions[]

    def explore(self) -> Edge:
        action = random.choice(self._possible_actions)
        if action not in self._explored_actions:
            state, cost = self._simulate(self._state, action)
            if state == self._state:
                next_node = self
            else:
                next_node = type(self)(state, self._simulate)
            self._explored_actions[action] = Edge(next_node, action, cost)
        return self._explored_actions[action]

    def update_best_edge(self, edge: Edge) -> bool:
        """
        Returns true if the given action leads to an updated best edge.
        """
        cost_to_next = edge.cost
        next_node: Node = edge.destination_node
        cost_to_go = cost_to_next + next_node.best_cost_to_go
        try:
            # In the beginning, we set best edge even without knowing
            # the actual costs to go of the next node.
            if (np.isinf(self._best_cost_to_go)
                    or cost_to_go < self._best_cost_to_go):
                self._best_edge = edge
                self._best_cost_to_go = cost_to_go
                return True
        except AttributeError:  # best_edge not set yet
            self._best_edge = edge
            self._best_cost_to_go = cost_to_go
            return True
        return False


class Edge:
    _destination_node: Node
    _action: ActionBase
    _cost: float

    def __init__(self, destination_node: Node, action: ActionBase,
                 cost: float):
        self._destination_node = destination_node
        self._action = action
        self._cost = cost

    @property
    def destination_node(self) -> Node:
        return self._destination_node

    @property
    def action(self) -> ActionBase:
        return self._action

    @property
    def cost(self) -> float:
        return self._cost
