from __future__ import annotations

from collections.abc import Iterable
import numpy as np
import random
from typing import Any, Callable

import configuration
from platoon_functionalities import graph_explorer, graph_tools, vehicle_platoon
import platoon_functionalities.platoon_lane_change_strategies as lc_strategies
import vehicle_group as vg
import vehicle_models.base_vehicle as base
import vehicle_models.four_state_vehicles as fsv


QuantizedState = configuration.QuantizedState
Command = tuple[int, set[int]]


# TODO: the training :)
# def train(initial_state: State, problem_map: ProblemMap, max_episodes: int,
#           epsilon: float = 0.5):#
#     sim_fun = problem_map.simulate
#     initial_node = Node2D(initial_state, sim_fun)
#     for episode in range(max_episodes):
#         if (episode / max_episodes * 100) % 10 == 0:
#             print(f'Episode: {episode / max_episodes * 100:.2f}%')
#             print(f'Best cost to go: {initial_node.best_cost_to_go}')
#         # if episode / max_episodes >= 0.5:
#         #     self._epsilon = self._default_epsilon
#
#         current_node = initial_node
#         path = []
#         while not problem_map.is_goal_state(current_node.state):
#             if current_node.is_new():
#                 current_node.generate_possible_actions()
#             if random.uniform(0, 1) < epsilon:  # explore
#                 edge = current_node.explore()
#             else:  # exploit
#                 edge = current_node.exploit()
#             # Save
#             path.append((current_node, edge))
#             # Update
#             current_node.update_best_edge(edge)
#             # Advance
#             current_node = edge.destination_node
#         current_node.set_as_terminal()
#         graph_explorer.update_along_path(path)
#     print(initial_node.best_cost_to_go)


class TrafficStateNode(graph_explorer.Node):
    _possible_actions: list[Command]
    _explored_actions: dict[Command, Edge]
    _simulate: Callable[[QuantizedState, Command], TrafficStateNode]

    _n_states = 4

    def __init__(self, state: QuantizedState,
                 sim_fun: Callable[[QuantizedState, Command],
                                   tuple[QuantizedState, float]],
                 tracker: graph_tools.PlatoonLCTracker):
        super().__init__(state, sim_fun)
        state_quantizer = graph_tools.StateQuantizer()
        self._continuous_state = state_quantizer.dequantize_state(state)
        self._tracker = tracker
        self._n_platoon = tracker.get_platoon_size()

    @staticmethod
    def order_values(lo_value: Any, platoon_value: Iterable[Any],
                     ld_value: Any) -> np.ndarray:
        """
        Used to ensure any stacked vectors always places the vehicles in the
        same order
        :param lo_value:
        :param platoon_value:
        :param ld_value:
        :return:
        """
        platoon_value_array = np.array(platoon_value).flatten()
        return np.hstack((lo_value, platoon_value_array, ld_value))

    def generate_possible_actions(self):
        initial_state = self._continuous_state
        vehicle_group = self._create_vehicle_group()
        vehicle_group.set_vehicles_initial_states_from_array(initial_state)
        vehicle_group.prepare_to_start_simulation(1)
        [veh.set_lane_change_direction(1) for veh
         in vehicle_group.vehicles.values()
         if (veh.is_in_a_platoon() and veh.get_current_lane() == 0)]
        vehicle_group.update_surrounding_vehicles()
        remaining_vehicles = sorted(self._tracker.remaining_vehicles)

        next_maneuver_steps: list[tuple[int, set[int]]] = []
        for next_pos_to_coop in self._tracker.get_possible_cooperative_vehicles():
            if next_pos_to_coop < 0:
                first_maneuver_steps = (
                    self._determine_next_maneuver_steps_no_coop(
                        vehicle_group, remaining_vehicles)
                )
                next_maneuver_steps |= first_maneuver_steps
                # Debugging
                # analysis.plot_state_vector(
                #     vehicle_group.get_full_initial_state_vector())
                # print(next_maneuver_steps)
            else:
                for i in range(len(remaining_vehicles)):
                    next_positions_to_move = set()
                    for j in range(i, len(remaining_vehicles)):
                        next_positions_to_move = (next_positions_to_move
                                                  | {remaining_vehicles[j]})
                        next_maneuver_steps.append(
                            (next_pos_to_coop, next_positions_to_move.copy()))
        return next_maneuver_steps

    def _determine_next_maneuver_steps_no_coop(
            self, vehicle_group: vg.VehicleGroup,
            remaining_vehicles: list[int]) -> list[tuple[int, set[int]]]:
        """
        When there is no cooperation, we only allow steps including vehicles
        that are already at safe position or that can perform a "quick"
        adjustment to be safe. It is the caller's duty to check whether there
        is cooperation.
        :return:
        """
        next_maneuver_steps: list[tuple[int, set[int]]] = []
        next_pos_to_coop = -1  # the caller should be sure of this
        for i in range(len(remaining_vehicles)):
            front_most_next_pos_to_move = remaining_vehicles[i]
            front_most_vehicle = (
                vehicle_group.get_platoon_vehicle_by_position(
                    front_most_next_pos_to_move)
            )
            ld = vehicle_group.get_vehicle_by_name("ld")
            if front_most_vehicle.get_x() <= ld.get_x():
                front_most_vehicle.check_surrounding_gaps_safety(
                    vehicle_group.vehicles,
                    ignore_orig_lane_leader=True)
                if front_most_vehicle.get_is_lane_change_safe():
                    next_positions_to_move = set()
                    for j in range(i, len(remaining_vehicles)):
                        next_positions_to_move = (
                                next_positions_to_move
                                | {remaining_vehicles[j]}
                        )
                        next_maneuver_steps.append(
                            (next_pos_to_coop, next_positions_to_move))
                    # We don't need to include cases that do not include
                    # the current front most vehicle if it is at a safe
                    # position
                    break
                else:
                    next_maneuver_steps.append(
                        (next_pos_to_coop,
                         {front_most_next_pos_to_move})
                    )
        return next_maneuver_steps

    def _create_vehicle_group(self, include_ld: bool = True
                              ) -> vg.ShortSimulationVehicleGroup:
        base.BaseVehicle.reset_vehicle_counter()
        lo = fsv.ShortSimulationVehicle(False)
        lo.set_name("lo")
        platoon_vehicles = []
        for i in range(self._n_platoon):
            veh = fsv.ShortSimulationVehicle(True, is_connected=True)
            veh.set_name("p" + str(i + 1))
            platoon_vehicles.append(veh)

        leader_platoon = vehicle_platoon.ClosedLoopPlatoon(
            platoon_vehicles[0], lc_strategies.StrategyMap.template)
        platoon_vehicles[0].set_platoon(leader_platoon)
        for i in range(1, len(platoon_vehicles)):
            leader_platoon.append_vehicle(platoon_vehicles[i])
            platoon_vehicles[i].set_platoon(leader_platoon)

        if include_ld:
            ld = fsv.ShortSimulationVehicle(False)
            ld.set_name("ld")
            leader_platoon._dest_lane_leader_id = ld.get_id()
        else:
            ld = []
        all_vehicles = self.order_values(lo, platoon_vehicles, ld)
        vehicle_group = vg.ShortSimulationVehicleGroup()
        vehicle_group.fill_vehicle_array(all_vehicles)
        vehicle_group.set_platoon_lane_change_strategy(
            lc_strategies.StrategyMap.template)

        return vehicle_group


class Edge(graph_explorer.Edge):
    _action: Command

    def __init__(self, destination_node: TrafficStateNode, action: Command,
                 cost: float):
        super().__init__(destination_node, action, cost)

    @property
    def action(self) -> Command:
        return self._action

