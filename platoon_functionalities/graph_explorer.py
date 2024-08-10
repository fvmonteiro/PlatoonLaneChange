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
        epsilon: float = 0.5, visited_nodes: dict[State, Node] = None,
        verbose_level: int = 0
) -> dict[str, list]:
    """
    Basic training script.
    :param initial_node: The source node
    :param max_episodes: The maximum number of episodes for training
    :param terminal_check_function: Function that determines whether a node is
     terminal
    :param epsilon: Epsilon for epsilon-greedy policy
    :param visited_nodes: A dictionary of visited nodes helps prevent creating
     repeated nodes.
    :param verbose_level: If 0, only prints the final cost to go. If 1, prints
     the best results every 10% done. If 2, also prints the explored path at
     every episode
    :returns: The best results per episode, which include the best path and
     best cost found so far.
    """

    if visited_nodes is None:
        visited_nodes = dict()
    visited_nodes[initial_node.state] = initial_node
    # if path_type is None:
    #     path_type = Path

    best_path = Path(initial_node)
    results = {"cost": [], "best_path": [], "time": []}
    start_time = time.time()
    for episode in range(max_episodes):
        current_node = initial_node
        path = Path(initial_node)
        while (not terminal_check_function(current_node)
               and path.cost < initial_node.best_cost_to_go):
            if (current_node.is_new()
                    or current_node.is_best_edge_fully_explored()
                    or random.uniform(0, 1) < epsilon):
                edge, next_node = current_node.explore(visited_nodes)
            else:
                edge, next_node = current_node.exploit()
            path.add_edge(edge)
            if path.cost >= initial_node.best_cost_to_go:
                current_node.discard_edge(edge)
            current_node.update_best_edge(edge)
            current_node = next_node
            visited_nodes[current_node.state] = current_node
        path.update_nodes()
        best_cost = initial_node.best_cost_to_go
        if len(results["cost"]) < 1 or best_cost < results["cost"][-1]:
            best_path = initial_node.get_best_path()
            results["cost"].append(initial_node.best_cost_to_go)
            results["best_path"].append(best_path)
            results["time"].append(time.time() - start_time)

        if initial_node.is_fully_explored:
            print(f"Best solution found in {episode} episodes")
            break

        if (verbose_level > 0 and episode > 0
                and (episode/max_episodes*100) % 10 == 0):
            print(f"Episode: {episode / max_episodes * 100:.2f}%")
            print(f"Best cost to go: {best_cost:.2f}")
            print("Path:\n", best_path.to_string(), sep="")
        if verbose_level > 1:
            print(f"Episode: {episode}. Searched path:\n{path.to_string()}")
            print(f"Cost: {initial_node.best_cost_to_go}")
    if verbose_level > 0:
        terminal_node = best_path.get_terminal_node()
        print(f"{Node.count} created nodes")
        print(f"{len(visited_nodes)} visited nodes")
        print(f"Final cost-to-go from {initial_node.state} to "
              f"{terminal_node.state}: {initial_node.best_cost_to_go:.2f}")
    return results


class Node (ABC):

    _state: State
    _possible_actions: list[ActionBase]
    # This dict is needed to prevent simulating the same scenario twice
    _explored_actions: dict[ActionBase, (Edge, Node)]
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
            return self.get_destination_node(self._best_edge).is_fully_explored
        except AttributeError:  # no best_edge yet
            warnings.warn("Trying to read best edge too soon")
            return False

    def exploit(self) -> tuple[Edge, Node]:
        if len(self._explored_actions) == 0:
            raise RuntimeError("Trying to exploit a node without any explored "
                               "actions")
        return self.take_action(self._best_edge.action)

    def explore(self, visited_nodes: dict[State, Node]
                ) -> tuple[Edge, Node]:
        if self.is_new():
            self.generate_possible_actions()
        action = random.choice(self._possible_actions)
        return self.take_action(action, visited_nodes)

    def take_action(
            self, action: ActionBase, visited_nodes: dict[State, Node] = None
    ) -> tuple[Edge, Node]:
        if visited_nodes is None:
            visited_nodes = dict()

        if action not in self._explored_actions:
            next_state, cost = self._simulate(self._state, action)
            if next_state in visited_nodes:
                self.add_edge_from_node(visited_nodes[next_state], action,
                                        cost)
            else:
                self.add_edge_from_state(next_state, action, cost)
        return self._explored_actions[action]

    def follow_edge(self, edge: Edge) -> tuple[Edge, Node]:
        action = edge.action
        return self.take_action(action)

    def get_destination_node(self, edge: Edge) -> Node:
        return self._explored_actions[edge.action][1]

    def update_best_edge(self, edge: Edge) -> bool:
        """
        Returns true if the given edge leas to smaller cost than the current
        best edge
        """
        cost_to_next = edge.cost
        next_node: Node = self._explored_actions[edge.action][1]
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
        min_cost_action = random.choice(list(self._explored_actions.keys()))
        _, dest_node = self.take_action(min_cost_action)
        best_cost_to_go = dest_node.best_cost_to_go
        for action in self._explored_actions:
            _, dest_node = self.take_action(action)
            if dest_node.best_cost_to_go < best_cost_to_go:
                min_cost_action = action
        self._best_edge, _ = self.take_action(min_cost_action)

    def discard_edge(self, edge: Edge) -> None:
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
        next_node = self.get_destination_node(edge)
        # next_node = self._explored_actions[edge.action][1]
        if next_node.is_fully_explored:
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

    def get_best_path(self) -> Path:
        path = Path(self)
        current_node = self
        while not current_node._best_cost_to_go != 0:
            try:
                edge, next_node = current_node.exploit()
                path.add_edge(edge)
            except AttributeError:
                print(f"There's no path from node {self.state} to a "
                      f"terminal node.")
                break
            current_node = next_node
        return path

    def traverse_action_sequence(self, actions: Iterable[ActionBase]) -> float:
        node = self
        visited_nodes = {node.state: node}
        cost = 0
        for action in actions:
            edge, next_node = node.take_action(action, visited_nodes)
            cost += edge.cost
            node = next_node
        return cost

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
    # _destination_node: Node
    _action: ActionBase
    _cost: float

    def __init__(self, action: ActionBase, cost: float):
        self._action = action
        self._cost = cost

    @property
    def action(self) -> ActionBase:
        return self._action

    @property
    def cost(self) -> float:
        return self._cost

    def action_to_str(self) -> str:
        return str(self.action)


class Path:
    _edges: list[Edge]
    _cost = 0

    def __init__(self, root_node: Node):
        self._root_node = root_node
        self._edges = []

    @property
    def root_node(self) -> Node:
        return self._root_node

    @property
    def edges(self) -> list[Edge]:
        return self._edges

    @property
    def cost(self) -> float:
        return self._cost

    def add_edge(self, edge: Edge) -> None:
        self._edges.append(edge)
        self._cost += edge.cost

    def get_terminal_node(self) -> Node:
        node = self._root_node
        for e in self._edges:
            node = node.get_destination_node(e)
        return node

    def update_nodes(self):
        nodes = [self._root_node]
        for edge in self._edges:
            nodes.append(nodes[-1].get_destination_node(edge))

        for i in range(len(self._edges) - 1, -1, -1):
            nodes[i].update_best_edge(self._edges[i])
            nodes[i].update_is_explored(self._edges[i])

    def to_string(self):
        ret_str = ""
        if len(self._root_node.state_to_str()) > 10:
            sep = "\n"
        else:
            sep = " "

        node = self._root_node
        for edge in self._edges:
            ret_str += (f"{node.state_to_str()}{sep}"
                        f"-{edge.action_to_str()}->{sep}")
            node = node.get_destination_node(edge)
        ret_str += f"{node.state_to_str()}\n"
        return ret_str
