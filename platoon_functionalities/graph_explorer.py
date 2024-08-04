from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
import random
import time
from typing import Any, Callable
import warnings
import weakref

import numpy as np

import configuration

ActionBase = Any
State = configuration.QuantizedState


def train_base(
        initial_node: Node, max_episodes: int,
        terminal_check_function: Callable[[Any], bool],
        epsilon: float = 0.5, verbose_level: int = 0,
        visited_nodes: dict[State, Node] = None
) -> dict[str, list]:
    """
    Basic training script.
    :param initial_node: The source node
    :param max_episodes: The maximum number of episodes for training
    :param terminal_check_function: Function that determines whether a node is
     terminal
    :param epsilon: Epsilon for epsilon-greedy policy
    :param verbose_level: If 0, only prints the final cost to go. If 1, prints
     the best results every 10% done. If 2, also prints the explored path at
     every episode
    :returns: The best results per episode, which include the best path and
     best cost found so far.
    """

    results = {"cost": [], "best_path": [], "time": []}
    if visited_nodes is None:
        visited_nodes = dict()
    visited_nodes[initial_node.state] = initial_node
    best_path = []
    start_time = time.time()
    for episode in range(max_episodes):
        current_node = initial_node
        path = []
        path_cost = 0
        while (not terminal_check_function(current_node)
               and path_cost < initial_node.best_cost_to_go):
            if (current_node.is_new()
                    or current_node.is_best_edge_fully_explored()
                    or random.uniform(0, 1) < epsilon):
                edge = current_node.explore(visited_nodes)
            else:
                edge = current_node.exploit()
            path.append((current_node, edge))
            path_cost += edge.cost
            if path_cost >= initial_node.best_cost_to_go:
                current_node.discard_edge(edge)
            current_node.update_best_edge(edge)
            current_node = edge.destination_node
            visited_nodes[current_node.state] = current_node
        update_along_path(path)
        best_cost = initial_node.best_cost_to_go
        if len(results["cost"]) < 1 or best_cost < results["cost"][-1]:
            best_path = get_best_path(initial_node,
                                      terminal_check_function)
            results["cost"].append(initial_node.best_cost_to_go)
            results["best_path"].append(best_path)
            results["time"].append(time.time() - start_time)

        if initial_node.is_fully_explored:
            print(f"Graph fully explored in {episode} episodes")
            break

        if (verbose_level > 0 and episode > 0
                and (episode/max_episodes*100) % 10 == 0):
            print(f"Episode: {episode / max_episodes * 100:.2f}%")
            print(f"Best cost to go: {best_cost:.2f}")
            print("Path:\n", path_to_string(best_path), sep="")
        if verbose_level > 1:
            print(f"Episode: {episode}. Searched path:\n{path_to_string(path)}")
            print(f"Cost: {initial_node.best_cost_to_go}")
    if verbose_level > 0:
        final_best_path = results["best_path"][-1]
        terminal_node = final_best_path[-1][1].destination_node
        print(f"{Node.count} created nodes")
        print(f"{len(visited_nodes)} visited nodes")
        print(f"Final cost-to-go from {initial_node.state} to "
              f"{terminal_node.state}: {initial_node.best_cost_to_go:.2f}")
    return results


def update_along_path(path: Path) -> None:
    for node, edge in path[::-1]:
        node.update_best_edge(edge)
        node.update_is_explored(edge)


def get_best_path(node: Node, terminal_check_function: Callable[[Any], bool],
                  ) -> Path:
    path = []
    current_node = node
    while not terminal_check_function(current_node):
        try:
            path.append((current_node, current_node.best_edge))
        except AttributeError:
            print(f"There's no path from node {node.state} to a "
                  f"terminal node.")
            break
        current_node = current_node.best_edge.destination_node
    # path.append((current_node, None))
    return path


def path_to_string(path: Path):
    ret_str = ""
    if len(path[0][0].state_to_str()) > 10:
        sep = "\n"
    else:
        sep = " "

    for node, edge in path:
        ret_str += f"{node.state_to_str()}{sep}-{edge.action_to_str()}->{sep}"
    ret_str += f"{path[-1][1].destination_node.state_to_str()}\n"
    return ret_str


def traverse_action_sequence(initial_node: Node,
                             actions: Iterable[ActionBase]) -> float:
    node = initial_node
    visited_nodes = {node.state: node}
    cost = 0
    for action in actions:
        edge = node.take_action(action, visited_nodes)
        cost += edge.cost
        node = edge.destination_node
    return cost


class Node (ABC):

    _state: State
    _possible_actions: list[ActionBase]
    # This dict is needed to prevent simulating the same scenario twice
    _explored_actions: dict[ActionBase, Edge]
    _best_edge: Edge
    _simulate: Callable[[State, ActionBase], tuple[State, float]]

    count = 0

    def __init__(self, state: State,
                 sim_fun: Callable[[State, ActionBase], tuple[State, float]]):
        self._state = state
        self._possible_actions = []
        self._explored_actions = dict()
        self._best_cost_to_go = np.inf
        self._simulate = sim_fun
        self._is_fully_explored = False
        Node.count += 1

    @property
    def state(self) -> State:
        return self._state

    @property
    def best_edge(self) -> Edge:
        return self._best_edge

    @property
    def best_cost_to_go(self) -> float:
        return self._best_cost_to_go

    @property
    def is_fully_explored(self) -> bool:
        return self._is_fully_explored

    def is_new(self) -> bool:
        return len(self._possible_actions) == 0

    def is_best_edge_fully_explored(self) -> bool:
        try:
            return self._best_edge.destination_node.is_fully_explored
        except AttributeError:  # no best_edge yet
            warnings.warn("Trying to read best edge too soon")
            return False

    def exploit(self) -> Edge:
        if len(self._explored_actions) == 0:
            raise RuntimeError("Trying to exploit a node without any explored "
                               "actions")
        return self._best_edge

    def explore(self, visited_nodes: dict[State, Node]) -> Edge:
        if self.is_new():
            self.generate_possible_actions()
        action = random.choice(self._possible_actions)
        return self.take_action(action, visited_nodes)

    def take_action(self, action: ActionBase, visited_nodes: dict[State, Node]
                    ) -> Edge:
        if action not in self._explored_actions:
            next_state, cost = self._simulate(self._state, action)
            if next_state in visited_nodes:
                self.add_edge_from_node(visited_nodes[next_state], action,
                                        cost)
            else:
                self.add_edge_from_state(next_state, action, cost)
        return self._explored_actions[action]

    def update_best_edge(self, edge: Edge) -> bool:
        """
        Returns true if the given edge leas to smaller cost than the current
        best edge
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

    def find_best_edge(self) -> None:
        """
        Looks for the edge with the lowest cost to go among all explored edges
        and updates the best edge accordingly
        """
        min_cost_edge = min(self._explored_actions.values(),
                            key=lambda x: x.destination_node.best_cost_to_go)
        self._best_edge = min_cost_edge

    def discard_edge(self, edge: Edge) -> None:
        # self._discarded_edges.add(edge)
        try:
            self._possible_actions.remove(edge.action)
        except ValueError:
            # TODO: avoid
            pass
            # print(f"[discard edge] action {edge.action} already removed "
            #       f"from {self.state}")
        if len(self._possible_actions) == 0:
            self._is_fully_explored = True

    def update_is_explored(self, edge: Edge) -> None:
        if edge.destination_node.is_fully_explored:
            # TODO: is this too time-consuming?
            try:
                self._possible_actions.remove(edge.action)
            except ValueError:
                # TODO: avoid
                pass
                # print(f"[update_is_explored] action {edge.action} already "
                #       f"removed from {self.state}")
            if len(self._possible_actions) == 0:
                self._is_fully_explored = True

    def set_as_terminal(self) -> None:
        self._is_fully_explored = True
        self._best_cost_to_go = 0

    def state_to_str(self) -> str:
        return str(self.state)

    @abstractmethod
    def add_edge_from_node(self, successor: Node, action: ActionBase,
                           cost: float) -> None:
        pass

    @abstractmethod
    def generate_possible_actions(self) -> None:
        pass

    @abstractmethod
    def add_edge_from_state(self, successor_state: State, action: ActionBase,
                            cost: float) -> None:
        pass

    def __eq__(self, other):
        return self.state == other.state

    def __repr__(self):
        return str(self.state)

    def __str__(self):
        return (f"Node: {self._state} with {len(self._possible_actions)} "
                f"possible actions and {len(self._explored_actions)} "
                f"explored actions")


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

    def action_to_str(self) -> str:
        return str(self.action)


Path = list[tuple[Node, Edge]]
