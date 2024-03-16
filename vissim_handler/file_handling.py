import warnings
from dataclasses import dataclass
import os
import shutil
from typing import Union

from vissim_handler.vissim_vehicle import VehicleType, PlatoonLaneChangeStrategy
from vissim_handler.scenario_handling import ScenarioInfo


@dataclass
class _PCInfo:
    """Used to help run simulation in different computers"""
    easy_id: str
    networks_folder: str
    shared_folder: str
    moves_folder: str
    moves_database_port: int


@dataclass
class _NetworkInfo:
    """Contains information about different VISSIM network"""
    name: str
    file_name: str
    n_lanes: int
    main_links: list[int]


_folders_map: dict[str, _PCInfo] = {
    "FEIJAO": _PCInfo("old_personal_pc",
                      "C:\\Users\\fvall\\Documents\\Research\\"
                      "TrafficSimulation\\VISSIM_networks",
                      "G:\\My Drive\\Safety in Mixed Traffic"
                      "\\data_exchange",
                      "C:\\Users\\fvall\\Documents\\Research\\"
                      "EnvironmentalEvaluations",
                      3307
                      ),
    "DESKTOP-P2O85S9": _PCInfo("old_personal_pc",
                               "C:\\Users\\fvall\\Documents\\Research\\"
                               "TrafficSimulation\\VISSIM_networks",
                               "G:\\My Drive\\Safety in Mixed Traffic"
                               "\\data_exchange",
                               "C:\\Users\\fvall\\Documents\\Research\\"
                               "EnvironmentalEvaluations",
                               3307
                               ),
    "DESKTOP-626HHGI": _PCInfo("usc-old",
                               "C:\\Users\\fvall\\Documents\\Research\\"
                               "AV_TrafficSimulation\\VISSIM_networks",
                               "C:\\Users\\fvall\\Google Drive\\"
                               "Safety in Mixed Traffic\\data_exchange",
                               "C:\\Users\\fvall\\Documents\\Research\\"
                               "EnvironmentalEvaluations",
                               3306
                               ),
    "DESKTOP-B1GECOE": _PCInfo("usc",
                               "C:\\Users\\fvall\\Documents\\Research\\"
                               "TrafficSimulation\\VISSIM_networks",
                               "G:\\My Drive\\Safety in Mixed Traffic"
                               "\\data_exchange",
                               "C:\\Users\\fvall\\Documents\\Research\\"
                               "EnvironmentalEvaluations",
                               3306
                               ),
}

# TODO: check main links of remaining scenarios
_network_info_all: dict[str, _NetworkInfo] = {
    "in_and_out":
        _NetworkInfo("in_and_out", "highway_in_and_out_lanes", 3, [3]),
    "platoon_discretionary_lane_change":
        _NetworkInfo("platoon_discretionary_lane_change",
                     "platoon_discretionary_lane_change", 2, [1, 3]),
}


def delete_files_in_folder(folder) -> None:
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


class FileHandler:
    """Class is the interface between scenario names and all their properties"""

    def __init__(self, scenario_name: str, is_data_in_cloud: bool = False):
        self.scenario_name = scenario_name
        if scenario_name.startswith("in_and_out"):
            self._network_info = _network_info_all["in_and_out"]
            # Get the string after "in_and_out"
            self.simulation_output_folder = scenario_name[len("in_and_out_"):]
        else:
            self._network_info = _network_info_all[scenario_name]
            self.simulation_output_folder = "results"
        # self._scenario_info = _scenario_info_all[scenario_name]
        self._is_data_in_cloud = is_data_in_cloud

    def set_is_data_in_cloud(self, use_cloud_directory: bool = False) -> None:
        self._is_data_in_cloud = use_cloud_directory

    def get_network_name(self) -> str:
        return self._network_info.name

    def get_file_name(self) -> str:
        return self._network_info.file_name

    def get_network_file_relative_address(self) -> str:
        # The network files are in folder with the same name as the file
        return self._network_info.file_name

    def get_networks_folder(self) -> str:
        if self._is_data_in_cloud:
            return get_cloud_networks_folder()
        else:
            return get_local_networks_folder()

    def get_network_file_folder(self) -> str:
        return os.path.join(self.get_networks_folder(),
                            self.get_network_file_relative_address())

    def get_results_base_folder(self) -> str:
        return os.path.join(self.get_networks_folder(),
                            self.get_results_relative_address())

    def get_main_links(self) -> list[int]:
        return self._network_info.main_links

    def get_n_lanes(self) -> int:
        return self._network_info.n_lanes

    def get_results_relative_address(self) -> str:
        return os.path.join(self.get_network_file_relative_address(),
                            self.simulation_output_folder)

    def get_moves_default_data_folder(self) -> str:
        return os.path.join(get_moves_folder(),
                            self.get_network_file_relative_address())

    def get_vissim_test_folder(self) -> str:
        return os.path.join(self.get_results_base_folder(), "test")

    def get_vissim_data_folder(self, scenario_info: ScenarioInfo) -> str:
        """
        Creates a string with the full path of the VISSIM simulation results
        data folder.
        """

        if ("in_and_out" in self.get_file_name()
                and scenario_info.vehicle_percentages is not None
                and sum(scenario_info.vehicle_percentages.values()) == 0):
            base_folder = self.get_network_file_folder()
        else:
            base_folder = self.get_results_base_folder()

        return create_file_path(base_folder, scenario_info)

    def get_moves_data_folder(self, scenario_info: ScenarioInfo) -> str:
        """
        Creates a string with the full path of the MOVES data
        folder.
        """
        return create_file_path(
            self.get_moves_default_data_folder(), scenario_info)

    def find_min_max_file_number(
            self, data_identifier: str, file_format: str,
            scenario_info: ScenarioInfo) -> tuple[int, int]:
        """"
        Looks for the file with the highest simulation number. This is
        usually the file containing results from all simulations.

        :param data_identifier: last part of the file name
        :param file_format: file extension
        :param scenario_info: Simulation scenario parameters
        :return: (min, max) simulation number.
        """
        max_simulation_number = -1
        min_simulation_number = 10000

        results_folder = self.get_vissim_data_folder(scenario_info)
        network_file = self.get_file_name()
        for file in os.listdir(results_folder):
            file_str = os.fsdecode(file)
            if (file_str.startswith(network_file + data_identifier)
                    and file_str.endswith(file_format)):
                file_no_extension = file_str.split(".")[0]
                try:
                    sim_number = int(file_no_extension.split("_")[-1])
                except ValueError:
                    print("File {} is not being read because its name does not "
                          "end with a number.".format(file_no_extension))
                    continue
                if sim_number > max_simulation_number:
                    max_simulation_number = sim_number
                if sim_number < min_simulation_number:
                    min_simulation_number = sim_number

        if max_simulation_number == -1:
            file_base_name = network_file + data_identifier + file_format
            raise FileNotFoundError("File {} not found in directory {}"
                                    .format(file_base_name, results_folder))

        return min_simulation_number, max_simulation_number

    def export_multiple_results_to_cloud(self, scenarios: list[ScenarioInfo]
                                         ) -> None:
        self._move_multiple_results(True, scenarios)

    def import_multiple_results_from_cloud(
            self, scenarios: list[ScenarioInfo]) -> None:
        self._move_multiple_results(False, scenarios)

    def _move_multiple_results(
            self, is_exporting: bool, scenarios: list[ScenarioInfo]) -> None:
        for sc in scenarios:
            try:
                self.move_result_files(is_exporting, sc)
            except FileNotFoundError:
                destination = "cloud" if is_exporting else "local"
                print("Couldn't move scenario {} to {} "
                      "folder.".format(sc, destination))
                continue

    def move_result_files(self, is_exporting: bool,
                          scenario_info: ScenarioInfo) -> None:
        """
        Moves data collections, link segments, and all post-processed data
        from the local to cloud folder is is_exporting is true, or from cloud
        to local is is_exporting is false.
        """

        temp = self._is_data_in_cloud
        if is_exporting:
            self.set_is_data_in_cloud(False)
            target_base = get_cloud_networks_folder()
            source_base = get_local_networks_folder()
        else:
            self.set_is_data_in_cloud(True)
            target_base = get_local_networks_folder()
            source_base = get_cloud_networks_folder()

        source_dir = self.get_vissim_data_folder(scenario_info)
        target_dir = os.path.join(target_base,
                                  source_dir.split(source_base + "\\")[1])
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        all_file_names = os.listdir(source_dir)
        all_csv_files = [file for file in all_file_names if
                         file.endswith("csv")]
        try:
            _, max_file_number = self.find_min_max_file_number(
                "", "att", scenario_info)
        except FileNotFoundError:
            print("No data collection measurements or link evaluation results"
                  " files found.")
            max_file_number = 0
        max_att_files = [file for file in all_file_names if
                         file.endswith(str(max_file_number) + ".att")]
        files_to_copy = all_csv_files + max_att_files
        for file_name in files_to_copy:
            shutil.copy(os.path.join(source_dir, file_name), target_dir)

        self.set_is_data_in_cloud(temp)

    def get_temp_results_folder(self) -> str:
        return os.path.join(self.get_networks_folder(), "temp_results")

    def copy_all_files_from_temp_folder(self, target_dir) -> None:
        source_dir = self.get_temp_results_folder()
        all_files = os.listdir(source_dir)
        for file_name in all_files:
            shutil.copy(os.path.join(source_dir, file_name), target_dir)


def create_percent_folder_name(
        vehicle_percentages: dict[VehicleType, int]) -> str:
    """Creates the name of the folder which contains results for the
    given percentage of controlled vehicles (not the full path)"""

    if sum(vehicle_percentages.values()) == 0:
        return "0_percent_"
    all_strings = []
    for vt, p in vehicle_percentages.items():
        if p > 0:
            all_strings += [str(int(p)), "percent", vt.name.lower()]
    return "_".join(all_strings)


def get_local_networks_folder() -> str:
    return _folders_map[os.environ["COMPUTERNAME"]].networks_folder


def get_cloud_networks_folder() -> str:
    return _folders_map[os.environ["COMPUTERNAME"]].shared_folder


def get_moves_folder() -> str:
    return _folders_map[os.environ["COMPUTERNAME"]].moves_folder


def get_moves_database_port() -> int:
    return _folders_map[os.environ["COMPUTERNAME"]].moves_database_port


def create_vehs_per_lane_folder_name(vehicles_per_lane: Union[int, str]) -> str:
    """Creates the name of the folder which contains results for the
    given vehicle per lane input (not the full path)"""

    if vehicles_per_lane is None:
        return ""
    if isinstance(vehicles_per_lane, str):
        vehs_per_lane_folder = vehicles_per_lane
    else:
        vehs_per_lane_folder = str(int(vehicles_per_lane)) + "_vehs_per_lane"
    return vehs_per_lane_folder


def create_platoon_lc_strategy_folder_name(
        platoon_lc_strategy: PlatoonLaneChangeStrategy) -> str:
    return platoon_lc_strategy.name


def create_speeds_folder_name(orig_and_dest_lane_speeds: tuple[int, str]
                              ) -> str:
    return "_".join(["origin_lane", str(orig_and_dest_lane_speeds[0]),
                     "dest_lane", str(orig_and_dest_lane_speeds[1])])


def create_file_path(
        base_folder: str, scenario_info: ScenarioInfo) -> str:
    """
    Creates a string with the full path of the data
    folder.

    :param base_folder: full path of the base folder where the data is
    :param scenario_info: Simulation scenario parameters
    :return: string with the folder where the data is
    """
    if (scenario_info.vehicle_percentages is None
            and scenario_info.vehicles_per_lane is None):
        warnings.warn("Using create_file_path to obtain the test folder"
                      " is deprecated. Use get_vissim_test_folder instead.")
        return os.path.join(base_folder, "test")

    folder_list = [base_folder]
    if scenario_info.platoon_lane_change_strategy is not None:
        folder_list.append(create_platoon_lc_strategy_folder_name(
            scenario_info.platoon_lane_change_strategy))
    if scenario_info.vehicles_per_lane > 0:
        folder_list.append(create_percent_folder_name(
            scenario_info.vehicle_percentages))
        folder_list.append(create_vehs_per_lane_folder_name(
            scenario_info.vehicles_per_lane))
        if scenario_info.orig_and_dest_lane_speeds is not None:
            folder_list.append(create_speeds_folder_name(
                scenario_info.orig_and_dest_lane_speeds))
    else:
        folder_list.append(create_vehs_per_lane_folder_name(
            scenario_info.vehicles_per_lane))
    if scenario_info.special_case is not None:
        folder_list.append(scenario_info.special_case)

    return os.path.join(*folder_list)
