from collections.abc import Iterable
import random

import networkx as nx


State = tuple[int, int]
Action = tuple[int, int]


class QLearningAgent:
    # Initialize parameters
    _default_alpha = 0.1  # Learning rate
    _default_gamma = 0.9  # Discount factor
    _default_epsilon = 0.1  # Exploration rate
    _actions: list[Action] = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    def __init__(
            self, grid_size: tuple[int, int], goal_nodes: Iterable[State],
            obstacles: set[State],
            alpha: float = _default_alpha, gamma: float = _default_gamma,
            epsilon: float = _default_epsilon):
        self._grid_size = grid_size
        self._obstacles = obstacles
        self._goal_nodes = goal_nodes
        self._alpha = alpha
        self._gamma = gamma
        self._epsilon = epsilon

        self._Q: dict[State, dict[Action, float]] = dict()
        self._graph = nx.DiGraph()
        for gn in goal_nodes:
            if not self._is_free_node(gn):
                raise ValueError(f'Goal node {gn} is not free')
            self._Q[gn] = {(0, 0): 0.}
            self._graph.add_node(gn)
            # self._graph.nodes[gn]['is_terminal'] = True

    def train(self, start_node: State, max_episodes: int):
        if not self._is_free_node(start_node):
            raise ValueError('Start node is not free')

        self._epsilon = 0.5
        for episode in range(max_episodes):
            if episode % 10 == 0:
                print(f'Episode: {episode/max_episodes*100:.2f}%')
            if episode / max_episodes >= 0.5:
                self._epsilon = self._default_epsilon

            current_node = start_node
            while not self._reached_sink(current_node):
                if current_node not in self._Q:
                    self._Q[current_node] = dict()
                    self._graph.add_node(current_node)

                if (len(self._Q[current_node]) == 0
                        or random.uniform(0, 1) < self._epsilon):
                    # Exploration: choose a random action (edge)
                    action = self._choose_random_action(current_node)
                    if action not in self._Q[current_node]:
                        next_node, cost = self._simulate(
                            current_node, action)
                        self._Q[current_node][action] = 0
                        if self._graph.has_edge(current_node, next_node):
                            self._graph[current_node][next_node][
                                "actions"].update({action})
                        else:
                            self._graph.add_edge(
                                current_node, next_node, weight=cost,
                                actions={action})
                else:
                    # Exploitation: choose the action with the highest Q-value
                    # among explored actions
                    action = self.explore(current_node)

                next_node, edge_cost = self._traverse_edge(current_node, action)
                reward = -edge_cost  # Since we are minimizing cost

                # Update Q-value
                # We can only choose among already explored actions
                if next_node not in self._Q or len(self._Q[next_node]) == 0:
                    best_cost_to_go = 0
                else:
                    best_cost_to_go = self._Q[next_node][
                        self.explore(next_node)]

                self._Q[current_node][action] += self._alpha * (
                        reward + self._gamma * best_cost_to_go
                        - self._Q[current_node][action])
                # Move to the next node
                current_node = next_node
            # print(self.print_Q())
        print(self.print_best_Q())

    def explore(self, current_node: State) -> Action:
        return max(self._Q[current_node], key=self._Q[current_node].get)

    def print_Q(self) -> str:
        ret_str = ''
        for state in sorted(self._Q.keys()):
            ret_str += f'{state} -> '
            for action in sorted(self._Q[state].keys()):
                ret_str += (f'{action}: '
                            f'{self._Q[state][action]:.1f}, ')
            ret_str += '\n'
        return ret_str

    def print_best_Q(self) -> str:
        ret_str = ''
        for i in range(self._grid_size[0]):
            for j in range(self._grid_size[1]):
                node = (i, j)
                if node in self._obstacles:
                    ret_str += '|xxxx|'
                elif node not in self._Q:
                    ret_str += '|    |'
                else:
                    ret_str += f'|{self._Q[node][self.explore(node)]:.1f}|'
            ret_str += '\n'
        return ret_str

    def _reached_sink(self, current_node: State) -> bool:
        return current_node in self._goal_nodes

    def _choose_random_action(self, current_node: State) -> Action:
        # available_actions = self._actions
        random_action = random.choice(self._actions)
        # new_state = (current_node[0] + random_action[0],
        #              current_node[1] + random_action[1])
        # while not (0 <= new_state[0] < self._grid_size[0]
        #            and 0 <= new_state[1] < self._grid_size[1]):
        #     available_actions.remove(random_action)
        #     random_action = random.choice(available_actions)
        #     new_state = (current_node[0] + random_action[0],
        #                  current_node[1] + random_action[1])
        return random_action

    def _simulate(self, current_node: State, action: Action
                  ) -> tuple[State, float]:
        new_node = (current_node[0] + action[0],
                    current_node[1] + action[1])
        if not self._is_free_node(new_node):
            cost = 1.e6
            # if new_node not in self._obstacles:  # allow traversing obstacles
            new_node = current_node
        else:
            cost = 1.
        return new_node, cost

    def _traverse_edge(self, current_node: State, action: Action
                       ) -> tuple[State, float]:
        for edge in self._graph.out_edges(current_node):
            if action in self._graph.edges[edge]["actions"]:
                return edge[1], self._graph.edges[edge]["weight"]
        print("oops")

    def _is_free_node(self, node: State) -> bool:
        return (0 <= node[0] < self._grid_size[0]
                and 0 <= node[1] < self._grid_size[1]
                and node not in self._obstacles)
