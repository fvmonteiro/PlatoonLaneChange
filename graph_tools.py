import copy
from collections.abc import Mapping, Sequence
from collections import deque
from typing import Any, Union

import networkx as nx
import numpy as np

import analysis
import configuration
import vehicle_group as vg
import vehicle_models.base_vehicle as base
import vehicle_models.four_state_vehicles as fsv


class PlatoonLCTracker:
    """
    Supporting class to make it easier to keep track of which vehicles have
    already moved to the destination lane and which are still in the origin
    lane
    """

    def __init__(self, n_platoon):
        self.remaining_vehicles = set([i for i in range(n_platoon)])
        self.moved_vehicles = set()

    def __repr__(self):
        return (f'[{self.__class__.__name__}] remaining|moved = '
                f'{self.remaining_vehicles}|{self.moved_vehicles}')

    def move_vehicle(self, position_in_platoon):
        self.remaining_vehicles.remove(position_in_platoon)
        self.moved_vehicles.add(position_in_platoon)

    def bring_back_vehicle(self, position_in_platoon):
        self.moved_vehicles.remove(position_in_platoon)
        self.remaining_vehicles.add(position_in_platoon)


class VehicleStatesGraph:

    def __init__(self, n_platoon: int):
        self.state_quantizer = StateQuantizer(10, 2)
        self.n_platoon = n_platoon

        # veh_type = fsv.ShortSimulationVehicle
        # lo = veh_type(False, False)
        # lo.set_name('lo')
        # platoon_vehs = []
        # for i in range(n_platoon):
        #     veh = veh_type(True, True)
        #     veh.set_name('p' + str(i + 1))
        #     platoon_vehs.append(veh)
        # ld = veh_type(False, False)
        # ld.set_name('ld')
        # all_vehicles = self.order_values(lo, platoon_vehs, ld)
        # self.vehicle_group: vg.VehicleGroup = vg.VehicleGroup()
        # self.vehicle_group.fill_vehicle_array(all_vehicles)
        #
        # vehicle_names = self.order_values(
        #     'lo', ['p' + str(i) for i in range(1, self.n_platoon + 1)], 'ld')
        # self.vehicle_group.set_vehicle_names(vehicle_names)

    def create_vehicle_group(self):
        base.BaseVehicle.reset_vehicle_counter()
        veh_type = fsv.ShortSimulationVehicle
        lo = veh_type(False, False)
        lo.set_name('lo')
        platoon_vehs = []
        for i in range(self.n_platoon):
            veh = veh_type(True, True)
            veh.set_name('p' + str(i + 1))
            platoon_vehs.append(veh)
        ld = veh_type(False, False)
        ld.set_name('ld')
        all_vehicles = self.order_values(lo, platoon_vehs, ld)
        vehicle_group: vg.VehicleGroup = vg.VehicleGroup()
        vehicle_group.fill_vehicle_array(all_vehicles)

        vehicle_names = self.order_values(
            'lo', ['p' + str(i) for i in range(1, self.n_platoon + 1)], 'ld')
        vehicle_group.set_vehicle_names(vehicle_names)
        return vehicle_group

    def order_values(self, lo_value: Any, platoon_value: Any, ld_value: Any
                     ) -> np.ndarray:
        if np.isscalar(platoon_value):
            platoon_value = [platoon_value] * self.n_platoon
        if isinstance(lo_value, np.ndarray):
            return np.hstack((lo_value, platoon_value, ld_value))
        return np.array([lo_value] + platoon_value + [ld_value])

    @staticmethod
    def split_values(values: Sequence[Any]) -> dict[str, Any]:
        return {'lo': values[0], 'platoon': values[1:-1], 'ld': values[-1]}

    def create_graph(self):
        # One loop for first vehicle to move
        # One loop for each choice of next vehicle
        # One loop for each possible cooperating veh
        # One loop (somewhere) for speeds

        all_positions = set([i for i in range(self.n_platoon)])
        # TODO: this trusts that vehicles are created in a specific order
        platoon_veh_ids = [i for i in range(1, self.n_platoon + 1)]

        # Eventually iterate over these too
        delta_x = {'lo': 0., 'ld': 0.}
        possible_vel = 10  # np.arange(10, 20, 5)

        # Desired speeds
        v0_lo = possible_vel
        v0_ld = possible_vel
        v_ff_platoon = possible_vel * 1.2
        free_flow_speeds = self.order_values(v0_lo, v_ff_platoon, v0_ld)
        # self.set_desired_speeds(v0_lo, v0_ld, v_ff_platoon)

        # First, we create all possible root nodes
        vehicle_group = self.create_vehicle_group()
        nodes: deque[tuple[PlatoonLCTracker, tuple]] = deque()
        for first_pos_to_move in all_positions:
            tracker = PlatoonLCTracker(self.n_platoon)
            tracker.move_vehicle(first_pos_to_move)
            v0_platoon = [possible_vel] * self.n_platoon
            nodes.appendleft((tracker, self.create_root_node(
                vehicle_group, v0_lo, v0_platoon, v0_ld, delta_x,
                first_pos_to_move)))

        # Then, we explore the children of each node in BFS mode
        visited_states: set[tuple] = set()
        while len(nodes) > 0:
            tracker, quantized_x0 = nodes.pop()
            if (quantized_x0 in visited_states
                    or len(tracker.remaining_vehicles) == 0):
                continue
            visited_states.add(quantized_x0)
            initial_state = self.state_quantizer.dequantize_state(quantized_x0)
            print(tracker)
            print('x0=', np.array(initial_state).reshape(-1, 4).transpose())
            for next_pos_to_coop in tracker.moved_vehicles:
                for next_pos_to_move in tracker.remaining_vehicles:
                    vehicle_group = self.create_vehicle_group()
                    vehicle_group.set_verbose(False)
                    next_id_to_coop = vehicle_group.vehicles[
                        platoon_veh_ids[next_pos_to_coop]].get_id()
                    print('  Coop veh:', vehicle_group.vehicles[
                        platoon_veh_ids[next_pos_to_coop]].get_name())
                    next_id_to_move = vehicle_group.vehicles[
                        platoon_veh_ids[next_pos_to_move]].get_id()
                    print('  Next to move:', vehicle_group.vehicles[
                        platoon_veh_ids[next_pos_to_move]].get_name())
                    success = self.simulate_till_lane_change(
                        vehicle_group, free_flow_speeds, initial_state,
                        next_id_to_move, next_id_to_coop)
                    data = vehicle_group.to_dataframe()
                    analysis.plot_trajectory(data)
                    analysis.plot_platoon_lane_change(data)
                    if success:
                        next_quantized_state = (
                            self.state_quantizer.quantize_state(
                                vehicle_group.get_current_state())
                        )
                        next_tracker = copy.deepcopy(tracker)
                        next_tracker.move_vehicle(next_pos_to_move)
                        nodes.appendleft((next_tracker, next_quantized_state))
                    else:
                        print('### Failed ###')

    # def set_desired_speeds(self, v_orig_leader: float, v_dest_leader: float,
    #                        v_platoon: Union[float, Sequence[float]] = None):
    #     if v_platoon is None:
    #         v_platoon = 1.2 * max(v_orig_leader, v_dest_leader)
    #     v_ff = self.order_values(v_orig_leader, v_platoon, v_dest_leader)
    #     self.vehicle_group.set_free_flow_speeds(v_ff)

    def create_root_node(
            self, vehicle_group: vg.VehicleGroup, v0_lo: float,
            v0_platoon: Sequence[float], v0_ld: float,
            delta_x: Mapping[str, float], first_move_pos_in_platoon: int
    ) -> tuple:

        ref_gaps = vehicle_group.get_initial_desired_gaps(
            self.order_values(v0_lo, v0_platoon, v0_ld))
        x0_platoon = np.zeros(self.n_platoon)
        y0_platoon = np.zeros(self.n_platoon)
        # Loop goes from p_2 to p_N because p1's position is already set to zero
        for i in range(1, self.n_platoon):
            x0_platoon[i] = x0_platoon[i - 1] - ref_gaps[i]
        idx_p1 = 1  # platoon leader idx in the vehicle array
        idx_lc = first_move_pos_in_platoon + idx_p1  # idx in the vehicle array
        y0_platoon[first_move_pos_in_platoon] = configuration.LANE_WIDTH
        theta0_platoon = np.array([0.] * self.n_platoon)
        # Ahead of the platoon in origin lane
        x0_lo = x0_platoon[0] + ref_gaps[idx_p1] + delta_x['lo']
        y0_lo = 0
        theta0_lo = 0.
        # Ahead of the platoon in dest lane
        x0_ld = (x0_platoon[first_move_pos_in_platoon]
                 + ref_gaps[idx_lc] + delta_x['ld'])
        y0_ld = configuration.LANE_WIDTH
        theta0_ld = 0.

        # Get single column platoon states
        platoon_states = np.vstack((x0_platoon, y0_platoon, theta0_platoon,
                                    v0_platoon)).reshape(-1, order='F')
        lo_states = np.hstack((x0_lo, y0_lo, theta0_lo, v0_lo))
        ld_states = np.hstack((x0_ld, y0_ld, theta0_ld, v0_ld))
        return self.state_quantizer.quantize_state(
            self.order_values(lo_states, platoon_states, ld_states))

    def simulate_till_lane_change(
            self, vehicle_group: vg.VehicleGroup,
            free_flow_speeds: Sequence[float], initial_state: np.ndarray,
            next_to_move: int, next_to_cooperate: int):

        dt = 1.0e-2
        tf = 20.
        time = np.arange(0, tf + dt, dt)

        vehicle_group.set_free_flow_speeds(free_flow_speeds)
        vehicle_group.set_vehicles_initial_states_from_array(
            initial_state)
        vehicle_group.prepare_to_start_simulation(len(time))
        vehicle_group.update_surrounding_vehicles()

        lc_vehicle = vehicle_group.get_vehicle_by_id(next_to_move)
        coop_vehicle = vehicle_group.get_vehicle_by_id(next_to_cooperate)
        lc_vehicle.set_lane_change_direction(1)
        lc_vehicle.set_desired_dest_lane_leader_id(
            coop_vehicle.get_origin_lane_leader_id())
        coop_vehicle.set_incoming_vehicle_id(lc_vehicle.get_id())
        i = 0
        while i < len(time) - 1 and lc_vehicle.has_lane_change_intention():
            vehicle_group.simulate_one_time_step(time[i + 1])
            i += 1

        vehicle_group.truncate_simulation_history()
        success = not lc_vehicle.has_lane_change_intention()
        return success

    # def set_initial_states(self, platoon_states: np.ndarray,
    #                        lo_states: np.ndarray, ld_states: np.ndarray):
    #     for veh in self.vehicle_group.yield_vehicles_in_order():
    #         # veh_id = veh.get_id()
    #         veh_name = veh.get_name()
    #         if veh_name == 'lo':
    #             initial_state = lo_states
    #         elif veh_name == 'ld':
    #             initial_state = ld_states
    #         else:
    #             position_in_platoon = int(veh_name[1:]) - 1
    #             initial_state = platoon_states[:, position_in_platoon]
    #         veh.set_initial_state(full_state=initial_state)

    # def compute_next_node(self, platoon_states, lo_states, ld_states,
    #                       next_to_move: int, next_to_cooperate: int):
    #     success, vehicle_group = self.simulate_till_lane_change(
    #         platoon_states, lo_states, ld_states, next_to_move,
    #         next_to_cooperate)
    #     full_state = vehicle_group.get_current_state()
    #     next_node = self.state_quantizer.quantize_state(
    #         vehicle_group.get_current_state())
    #     time_cost = full_state[-1]


# class NodeCreator:
#     def __init__(self, n: int):
#         self.state_quantizer = StateQuantizer(10, 2)
#         self.sorted_vehicle_ids = [i for i in range(n)]
#
#     def make_node(self, vehicle_group: vg.VehicleGroup):
#         state = vehicle_group.get_current_state()


class StateQuantizer:
    """
    Class to manage quantization of states
    """

    # Possible changes for speed or maintainability:
    # - Transform everything in numpy arrays and try to vectorize the
    #  quantize and dequantize operations in case this becomes a bottleneck
    # - Make the methods work with vehicle objects if that's better
    # - Set n_vehicles in constructor

    def __init__(
            self, dx: float, dv: float, dy: float = 4, dtheta: float = None,
            veh_type: type[base.BaseVehicle] = fsv.FourStateVehicle):
        self.intervals = []
        self.min_value = []
        for state in veh_type.get_state_names():
            if state == 'x':
                self.intervals.append(dx)
                self.min_value.append(0.)
            elif state == 'y':
                self.intervals.append(dy)
                self.min_value.append(-2.)
            elif state == 'theta':
                self.intervals.append(dtheta if dtheta is not None else np.inf)
                self.min_value.append(-np.pi / 2)
            elif state == 'v':
                self.intervals.append(dv)
                self.min_value.append(0.)
            else:
                raise ValueError(f'Vehicle type {veh_type} has an unknown'
                                 f' state: {state}.')

    def quantize_state(self, full_system_state: Sequence[float]) -> tuple[int]:
        """
        Computes the quantized version of the system state (stack of all
        vehicle states)
        :param full_system_state:
        :return:
        """
        i = 0
        n = len(full_system_state)  # time is the last state
        qx = np.zeros(n, dtype=int)
        while i < n:
            for j in range(len(self.intervals)):
                qx[i] = ((full_system_state[i] - self.min_value[j])
                         // self.intervals[j])
                i += 1
        return tuple(qx)

    def dequantize_state(self, full_quantized_state: Sequence[float],
                         mode: str = 'mean') -> np.ndarray:
        """
        Estimates the continuous system state given the quantized state. The
        estimate is done based on the mode.
        :param full_quantized_state:
        :param mode:
        :return:
        """
        i = 0
        n = len(full_quantized_state)  # time is the last state
        x = np.zeros(n)
        if mode == 'min':
            delta = 0
        elif mode == 'mean':
            delta = 0.5
        elif mode == 'max':
            delta = 1.
        else:
            raise ValueError('Parameter mode must be "min", "mean", or "max"')

        while i < n:
            for j in range(len(self.intervals)):
                if self.intervals[j] == np.inf:
                    x[i] = 0.
                else:
                    x[i] = ((full_quantized_state[i] + delta)
                            * self.intervals[j] + self.min_value[j])
                i += 1
        return x
