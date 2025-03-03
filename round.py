import itertools

import numpy as np
import scipy.spatial
import shapely
from tinyflux import TinyFlux
from helpers import *
from fountain_metric import *
from round_plotter import RoundPlotter
from typing import Dict, Union
import os
import file_last_modification_time_finder
import alpha_shapes

import sklearn


class Round:

    def __init__(self,
                 file_path: str,
                 T_start: int = 0,
                 T: int = None,
                 n_agents: int = 50,
                 n_preds: int = 1,
                 center: Tuple[float, float] = (0., 0.),
                 radius: float = 20.,
                 width_in_m: float = 3.80,
                 remove_dup_timevals: bool = False,
                 infer_timestamps: bool = True,
                 write_timestamps_to_csv: bool = False,
                 dist_tolerance: float = 0.9,
                 margin: int = 0,
                 min_bout_length: int = 2,
                 speed_threshold: float = 0.,
                 speed_tolerance: float = 1.,
                 absolute_speed_threshold: float = None,
                 bout_ids_to_remove: Iterable[str] = None,
                 evasion_vel_angle_change_limit: float = 90.,
                 evasion_pred_dist_to_com_limit: float = 0.4
                 ):
        # Save experiment parameters
        self.file_path = file_path
        self.T_start = T_start
        self.T = T
        self.n_agents = n_agents
        self.n_preds = n_preds
        self.suffix = file_path.split('/')[-2][1]

        width = ((2 ** 0.5) * radius + radius * 2) / 2

        self.unit_space = width_in_m / (2 * radius)
        self.radius = radius * self.unit_space
        self.center = (center[0] * self.unit_space, center[1] * self.unit_space)
        self.width = width * self.unit_space
        print(f"Unit space: {self.unit_space}")

        self.arena_points = [(self.center[0] - self.width / 2, self.center[1] - self.width / 2),
                             (self.center[0] + self.width / 2, self.center[1] - self.width / 2),
                             (self.center[0] + self.width / 2, self.center[1] + self.width / 2),
                             (self.center[0] - self.width / 2, self.center[1] + self.width / 2)]

        # Initializing TinyFlux client
        print(f"Loading round data from {self.file_path}")
        assert os.path.exists(self.file_path)
        self.db = TinyFlux(self.file_path)

        # retrieving timestep data to determine recording order
        timesteps = self.db.get_field_values("ts")
        # normalizing timestep to start with 0
        timesteps = [t - min(timesteps) for t in timesteps]

        if T is None:
            T = len(timesteps)

        # retrieving timestamp data to determine time of each observation
        self.avg_fs = self.infer_fs()
        if infer_timestamps:
            self.timestamps = np.array(timesteps) / self.avg_fs
        else:
            self.timestamps = self.db.get_field_values("t")

        if write_timestamps_to_csv:
            write_column_to_csv(file_path=self.file_path, column=self.timestamps, column_name='f_t')

        self.timestamps = [t - min(self.timestamps) for t in self.timestamps]

        #check ctime ordering is the same as timesteps (but json files are more than rows of csv)
        #ctimes = infer_timestamps_from_json_files(self.file_path)
        #print(np.sum(np.array([ctimes[i] for i in sorted(range(len(timesteps)), key=lambda x: timesteps[x])]) != np.array([ctimes[i] for i in sorted(range(len(ctimes)), key=lambda x: ctimes[x])])))
        #assert [ctimes[i] for i in sorted(range(len(timesteps)), key=lambda x: timesteps[x])] == [ctimes[i] for i in sorted(range(len(ctimes)), key=lambda x: ctimes[x])]

        # retrieving data for a single agent
        self.agents_data = []
        self.agent_data_arrs = []

        for agent_id in range(self.n_agents):
            agent_x = self.db.get_field_values("x" + str(agent_id))
            agent_y = self.db.get_field_values("y" + str(agent_id))
            agent_data = [(agent_x[i] * self.unit_space + self.center[0], agent_y[i] * self.unit_space + self.center[1]) for i in range(len(timesteps))]
            # sorting agent data according to timesteps list
            agent_data = [agent_data[i] for i in sorted(range(len(timesteps)), key=lambda x: timesteps[x])]
            # slicing to length
            agent_data = agent_data[self.T_start:self.T]
            agent_data_arr = convert_list_of_tuples_to_array(agent_data)
            self.agents_data.append(agent_data)
            self.agent_data_arrs.append(agent_data_arr)

        # retrieving data for predator agent(s)
        self.preds_data = []
        self.pred_data_arrs = []

        for pred_id in range(self.n_preds):
            pred_x = self.db.get_field_values("prx" + str(pred_id))
            pred_y = self.db.get_field_values("pry" + str(pred_id))
            pred_data = [(pred_x[i] * self.unit_space + self.center[0], pred_y[i] * self.unit_space + self.center[1]) for i in range(len(timesteps))]
            # sorting pred data according to timesteps list
            pred_data = [pred_data[i] for i in sorted(range(len(timesteps)), key=lambda x: timesteps[x])]
            # slicing to length
            pred_data = pred_data[self.T_start:self.T]
            pred_data_arr = convert_list_of_tuples_to_array(pred_data)
            self.preds_data.append(pred_data)
            self.pred_data_arrs.append(pred_data_arr)

        # sorting and slicing timesteps and timestamps
        self.timesteps = np.array(sorted(timesteps)[T_start:T])
        self.timestamps = np.array(sorted(self.timestamps)[T_start:T])

        print(f"{np.sum(self.timestamps[1:] == self.timestamps[:-1])} duplicates in time detected")
        if remove_dup_timevals:
            print(f"Removing time duplicates")
            dup_filter = np.hstack([np.array([True]), self.timestamps[1:] != self.timestamps[:-1]])
            self.timesteps = self.timesteps[dup_filter]
            self.timestamps = self.timestamps[dup_filter]
            self.preds_data = [np.array(pred_data)[dup_filter] for pred_data in self.preds_data]
            self.pred_data_arrs = [np.array(pred_data_arr)[dup_filter] for pred_data_arr in self.pred_data_arrs]
            self.agents_data = [np.array(agent_data)[dup_filter] for agent_data in self.agents_data]
            self.agent_data_arrs = [np.array(agent_data_arr)[dup_filter] for agent_data_arr in self.agent_data_arrs]

        # storing agent center of mass over time
        self.agent_com = np.mean(np.array(self.agent_data_arrs), axis=0)

        # create bouts
        self.pred_bout_bounds = None
        self.pred_bout_lengths = None
        self.pred_data_arrs_bouts = None
        self.pred_vel_arrs_bouts = None
        self.agent_data_arrs_bouts = None
        self.timestamps_bouts = None
        self.agent_com_bouts = None
        self.pred_bout_ids_filtered = None

        self.pred_bout_bounds_filtered = None
        self.pred_bout_bounds_discarded = []
        self.pred_bout_ids_discarded = []
        self.pred_data_arrs_bouts_discarded = None
        self.agent_data_arrs_bouts_discarded = None
        self.timestamps_bouts_discarded = None
        self.agent_com_bouts_discarded = None
        self.segment_into_bouts(dist_tolerance=dist_tolerance, margin=margin,
                                min_bout_length=min_bout_length,
                                speed_threshold=speed_threshold, speed_tolerance=speed_tolerance,
                                absolute_speed_threshold=absolute_speed_threshold,
                                bout_ids_to_remove=bout_ids_to_remove)
        self.compute_bout_evasion_bounds(vel_angle_change_limit=evasion_vel_angle_change_limit,
                                         pred_dist_to_com_limit=evasion_pred_dist_to_com_limit)

        print("Round Summary:")
        print(f"-> fs: {self.avg_fs}, dt = {1/self.avg_fs}")
        print(f"-> Start time: {self.timestamps[0]} s, End time: {self.timestamps[-1]} s")
        print(f"-> Total round duration: {self.timestamps[-1] - self.timestamps[0]} s")

    def infer_fs(self) -> float:

        round_identifier = self.file_path.split('/')[-2]
        json_file_folder = "/".join(self.file_path.split('/')[:-1] + [f'database_input{round_identifier}/'])
        if not os.path.exists(json_file_folder):
            json_file_folder = "/".join(self.file_path.split('/')[:-1] + [f'database_input{round_identifier.lower()}/'])

        if not os.path.exists(json_file_folder):
            round_identifier = self.file_path.split('/')[-1].split('.')[0].split('_')[-1]
            json_file_folder = "/".join(self.file_path.split('/')[:-1] + [f'database_input{round_identifier}/'])

        if not os.path.exists(json_file_folder):
            json_file_folder = "/".join(self.file_path.split('/')[:-1] + [f'database_input{self.file_path.split('/')[-2]}/'])

        if not os.path.exists(json_file_folder):
            round_identifier = self.file_path.split('/')[-1].split('.')[0]
            json_file_folder = "/".join(self.file_path.split('/')[:-1] + [f'database_input{''.join(round_identifier.split('_')[-2:]).lower()}/'])

        json_files = file_last_modification_time_finder.get_json_files(json_file_folder)
        print(f"Found {len(json_files)} json files in the folder.")

        # ordering files by filename
        json_files = file_last_modification_time_finder.sort_files_by_filename(json_files)

        json_files = [os.path.join(json_file_folder, file) for file in json_files]
        print(f"Files last modified between {file_last_modification_time_finder.file_last_modification_time_minutes_seconds(json_files[0])} and {file_last_modification_time_finder.file_last_modification_time_minutes_seconds(json_files[-1])}.")

        # getting the number of files that were in a given second
        _, num_files_between = file_last_modification_time_finder.find_consecutive_files_with_increasing_seconds(json_files)

        # calculated framerate in each minute
        framerates_per_min = [rate for rate in num_files_between]

        # removing first and last elements as these can be not full minutes
        framerates_per_min = framerates_per_min[1:-1]

        avg_fs = sum(framerates_per_min) / len(framerates_per_min)
        return avg_fs

    def compute_speed(self, data_arrs: List[NDArray[float]]) -> List[NDArray[float]]:
        return [np.insert(np.sqrt(np.sum(np.diff(data_arr, axis=0) ** 2, axis=1)) / (np.diff(self.timestamps)), 0, [0.,]) for data_arr in data_arrs]

    def compute_velocity(self, data_arrs: List[NDArray[float]]) -> List[NDArray[float]]:
        return [np.vstack([np.diff(data_arr, axis=0) / (np.diff(self.timestamps).reshape((-1, 1))), [0., 0.]]) for data_arr in data_arrs]

    def compute_predator_velocity(self):
        return self.compute_velocity(self.pred_data_arrs)

    def compute_agent_velocity(self):
        return self.compute_velocity(self.agent_data_arrs)

    def compute_agent_polarisation(self):
        agents_velocity = self.compute_agent_velocity()
        normalized_agents_velocity = [agent_velocity / np.linalg.norm(agent_velocity, axis=-1).reshape((-1, 1)) for agent_velocity in agents_velocity]
        normalized_agents_velocity_T = transpose_list_of_arrays(normalized_agents_velocity)
        return np.array([np.linalg.norm(np.mean(normalized_agent_velocity_T, axis=0)) for normalized_agent_velocity_T in normalized_agents_velocity_T])

    def compute_agent_speed(self) -> List[NDArray[float]]:
        agents_data_speed = self.compute_speed(self.agent_data_arrs)
        return agents_data_speed

    def compute_agent_com_speed(self) -> NDArray[float]:
        return self.compute_speed([self.agent_com])[0]

    def compute_predator_speed(self) -> List[NDArray[float]]:
        preds_speed = self.compute_speed(self.pred_data_arrs)
        return preds_speed

    def compute_predator_to_agent_com_speed_ratio(self) -> List[NDArray[float]]:
        agent_com_speed = self.compute_agent_com_speed()
        preds_speed = self.compute_predator_speed()
        return [preds_speed[pid]/agent_com_speed for pid in range(self.n_preds)]

    def compute_acceleration(self, datas_vel: List[NDArray[float]],
                             smooth: bool = True, smoothing_args: Dict = None) -> List[NDArray[float]]:
        datas_acc = [np.insert(np.diff(data_vel) / (np.diff(self.timestamps)), 0, [0., ]) for data_vel in datas_vel]
        if smooth:
            datas_acc = smooth_metric(data_arrs=datas_acc, smoothing_args=smoothing_args)

        return [np.insert(data_acc, 0, [0., ]) for data_acc in datas_acc]

    def compute_agent_acceleration(self, smooth: bool = True, smoothing_args: Dict = None) -> List[NDArray[float]]:
        agents_data_speed = self.compute_agent_speed()
        agents_data_acc = self.compute_acceleration(agents_data_speed, smooth, smoothing_args)

        return agents_data_acc

    def compute_predator_acceleration(self, smooth: bool = True, smoothing_args: Dict = None) -> List[NDArray[float]]:
        preds_data_speed = self.compute_predator_speed()
        preds_data_acc = self.compute_acceleration(preds_data_speed, smooth, smoothing_args)
        return preds_data_acc

    def compute_predator_distance_to_agents(self) -> List[NDArray[float]]:
        pred_dist_to_agents = [np.vstack([np.sqrt(np.sum((self.pred_data_arrs[pid] - self.agent_data_arrs[aid])**2, axis=1)) for aid in range(self.n_agents)]).T for pid in range(self.n_preds)]

        return pred_dist_to_agents

    def compute_predator_distance_to_agent_com(self) -> List[NDArray[float]]:
        pred_dist_to_agent_com = [np.sqrt(np.sum((pred_data_arr - self.agent_com)**2, axis=1)) for pred_data_arr in self.pred_data_arrs]

        return pred_dist_to_agent_com

    def compute_predator_distance_to_border(self) -> List[NDArray[float]]: # to implement
        pred_dist_to_border = [shortest_dist_to_polygon(self.arena_points, pred_data) for pred_data in self.pred_data_arrs]

        return pred_dist_to_border

    def compute_agent_com_distance_to_border(self) -> NDArray[float]:  # to implement
        agent_com_dist_to_border = shortest_dist_to_circle(self.center, self.radius, self.agent_com.T)

        return agent_com_dist_to_border

    def compute_predator_distance_to_center(self) -> List[NDArray[float]]:
        pred_dist_to_center = [euclidean_distance(np.array(self.center).reshape((-1,1)), pred_data_arr.T) for pred_data_arr in self.pred_data_arrs]

        return pred_dist_to_center

    def compute_predator_attack_angle(self, smooth: bool = False, smoothing_args: Dict = None) -> List[NDArray[float]]:
        preds_attack_angles = [compute_angle(self.agent_com[:-1], pred_data_arr[:-1], pred_data_arr[1]) for pred_data_arr in self.pred_data_arrs]
        if smooth:
            preds_attack_angles = smooth_metric(data_arrs=preds_attack_angles, smoothing_args=smoothing_args)

        return [np.insert(pred_attack_angles, len(pred_attack_angles), [0., ]) for pred_attack_angles in preds_attack_angles]

    def compute_preys_behind_predator(self):
        pred_vel_arrs = self.compute_predator_velocity()

        preys_behind = [np.zeros((len(self.timestamps), len(self.agents_data))) for _ in range(self.n_preds)]
        for pid in range(self.n_preds):
            for tid in range(len(self.timestamps)):
                for aid in range(len(self.agents_data)):
                    pred_vel_vec = pred_vel_arrs[pid][tid]
                    agent_pos_vec = np.array(self.agents_data[aid][tid]) - self.preds_data[pid][tid]
                    preys_behind[pid][tid, aid] = 1 if agent_pos_vec @ pred_vel_vec < 0 else 0

        return preys_behind

    def compute_n_preys_behind_predator(self):
        preys_behind = self.compute_preys_behind_predator()

        return [np.sum(preys_behind[pid], axis=-1) for pid in range(self.n_preds)]

    def check_which_side_of_predator_preys_on(self):
        pred_vel_arrs = self.compute_predator_velocity()

        preys_sides = [np.zeros((len(self.timestamps), len(self.agents_data))) for _ in range(self.n_preds)]
        for pid in range(self.n_preds):
            for tid in range(len(self.timestamps)):
                for aid in range(len(self.agents_data)):
                    pred_vel_vec = get_perpendicular_vector(pred_vel_arrs[pid][tid])[0]
                    agent_pos_vec = np.array(self.agents_data[aid][tid]) - self.preds_data[pid][tid]
                    preys_sides[pid][tid, aid] = 1 if agent_pos_vec @ pred_vel_vec < 0 else -1

        return preys_sides

    def check_if_preys_on_both_sides_of_predator(self) -> List[NDArray[bool]]:
        prey_sides = self.check_which_side_of_predator_preys_on()

        return [np.logical_and(np.sum(prey_sides[pid], axis=-1) != -self.n_agents, np.sum(prey_sides[pid], axis=-1) != self.n_agents) for pid in range(self.n_preds)]

    def compute_agent_distance_to_nearest_neighbor(self) -> List[NDArray[float]]:
        agent_positions = transpose_list_of_arrays(self.agent_data_arrs)
        agent_nearest_neighbors = []
        for agent_positions_arr in agent_positions:
            dist_matrix = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(agent_positions_arr, metric=euclidean_distance))

            agent_nearest_neighbors.append(np.apply_along_axis(lambda arr: np.nanmin([val if val != 0 else np.nan for val in arr]), 0, dist_matrix))

        return transpose_list_of_arrays(agent_nearest_neighbors)

    def segment_into_bouts(self,
                           dist_tolerance: float = 0.9,
                           margin: int = 0,
                           min_bout_length: int = 2,
                           speed_threshold: float = 0.,
                           speed_tolerance: float = 1.,
                           absolute_speed_threshold: float = None,
                           bout_ids_to_remove: List[str] = None):
        dist_threshold = dist_tolerance*self.width/2

        print(f"Segmenting bouts using threshold of {dist_threshold}, with a margin of {margin} observations, on predator distance from arena center {self.center}")
        pred_dist_to_center = self.compute_predator_distance_to_center()

        pred_points_during_bouts = [pred_dist_to_center < dist_threshold for pred_dist_to_center in pred_dist_to_center]
        temp = [get_bounds(pred_points_during_bouts, margin=margin) for pred_points_during_bouts in pred_points_during_bouts]
        self.pred_bout_bounds = [temp_el[0] for temp_el in temp]
        self.pred_bout_lengths = [temp_el[1] for temp_el in temp]
        self.pred_bout_ids = [np.arange(len(self.pred_bout_bounds[pid])).astype(str) + np.full(len(self.pred_bout_bounds[pid]), '_' + (self.suffix if self.n_preds == 1 else str(pid + 1) + "_SH")) for pid in range(self.n_preds)]
        self.pred_bout_ids_filtered = self.pred_bout_ids
        self.pred_bout_bounds_filtered = self.pred_bout_bounds

        print(f"--> found {np.sum([len(self.pred_bout_bounds[pid]) for pid in range(self.n_preds)])}")

        if min_bout_length is not None:
            self.apply_bout_length_filter(min_bout_length=min_bout_length)

        if speed_threshold is not None and speed_tolerance is not None:
            self.apply_bout_low_speed_filter(speed_threshold=speed_threshold, speed_tolerance=speed_tolerance)

        if absolute_speed_threshold is not None:
            self.apply_bout_high_speed_filter(absolute_speed_threshold=absolute_speed_threshold)

        if bout_ids_to_remove is not None:
            self.remove_bout_by_id(ids=bout_ids_to_remove)

        self.organize_bout_info()

    def filter_bouts(self, filters: Iterable[Iterable[bool]]):
        self.pred_bout_bounds_discarded.append([np.array(self.pred_bout_bounds_filtered[pid])[np.logical_not(filters[pid])] for pid in range(self.n_preds)])
        self.pred_bout_ids_discarded.append([self.pred_bout_ids_filtered[pid][np.logical_not(filters[pid])].tolist() for pid in range(self.n_preds)])

        self.pred_bout_bounds_filtered = [np.array(self.pred_bout_bounds_filtered[pid])[filters[pid]] for pid in range(self.n_preds)]
        self.pred_bout_ids_filtered = [self.pred_bout_ids_filtered[pid][filters[pid]] for pid in range(self.n_preds)]

        print(f"--> Discarded bouts: {self.pred_bout_ids_discarded[-1]}")

    def apply_bout_length_filter(self, min_bout_length: int):
        print(f"Discarding bouts with less than {min_bout_length} observations")

        pred_bout_length_filters = [np.array(self.pred_bout_lengths[pid]) >= min_bout_length for pid in
                                    range(self.n_preds)]

        self.filter_bouts(filters=pred_bout_length_filters)

        print(f"--> {np.sum([len(self.pred_bout_bounds_filtered[pid]) for pid in range(self.n_preds)])} remain")

    def apply_bout_low_speed_filter(self, speed_threshold: float, speed_tolerance: float):
        print(f"Discarding bouts with speed less than {speed_threshold} for more than {speed_tolerance*100}% of observations")
        preds_data_vel = self.compute_predator_speed()
        pred_bouts_vel = [[preds_data_vel[pid][self.pred_bout_bounds_filtered[pid][i][0]:self.pred_bout_bounds_filtered[pid][i][1]] for i in range(len(self.pred_bout_bounds_filtered[pid]))] for pid in range(self.n_preds)]
        pred_bout_speed_filters = [np.array([not np.sum(np.array(pred_bouts_vel[pid][i]) < speed_threshold)/len(pred_bouts_vel[pid][i]) > speed_tolerance for i in range(len(pred_bouts_vel[pid]))]) for pid in range(self.n_preds)]

        self.filter_bouts(filters=pred_bout_speed_filters)

        print(f"--> {np.sum([len(self.pred_bout_bounds_filtered[pid]) for pid in range(self.n_preds)])} remain")

    def apply_bout_high_speed_filter(self, absolute_speed_threshold: float):
        print(f"Discarding bouts with speed greater than {absolute_speed_threshold}")

        preds_data_vel = self.compute_predator_speed()
        pred_dists_to_com = self.compute_predator_distance_to_agent_com()
        pred_dists_to_border = self.compute_predator_distance_to_border()
        pred_bout_abs_speed_filters = [[] for _ in range(self.n_preds)]
        for pid in range(self.n_preds):
            for pred_bout_bound in self.pred_bout_bounds_filtered[pid]:
                pred_bouts_vel = [preds_data_vel[pid][pred_bout_bound[0]:pred_bout_bound[1]] for pid in
                                  range(self.n_preds)]

                pred_bout_abs_speed_filter = True

                for pid2 in range(self.n_preds):
                    if np.any(pred_bouts_vel[pid2] > absolute_speed_threshold):
                        invalid_val_ids = np.where(pred_bouts_vel[pid2] > absolute_speed_threshold)
                        if pid == pid2:
                            dist_filter = np.all(pred_dists_to_border[pid2][pred_bout_bound[0]:pred_bout_bound[1]][
                                                     invalid_val_ids] < 0.2 * self.width / 2)
                        else:
                            dist_filter = np.all(pred_dists_to_com[pid2][pred_bout_bound[0]:pred_bout_bound[1]][
                                                     invalid_val_ids] > 0.3 * self.width / 2)
                        pred_bout_abs_speed_filter = pred_bout_abs_speed_filter and dist_filter
                pred_bout_abs_speed_filters[pid].append(pred_bout_abs_speed_filter)

        self.filter_bouts(filters=pred_bout_abs_speed_filters)

        print(f"--> {np.sum([len(self.pred_bout_bounds_filtered[pid]) for pid in range(self.n_preds)])} remain")

    def remove_bout_by_id(self, ids: List[str]):
        pred_bout_ids_filters = [np.logical_not(np.isin(self.pred_bout_ids_filtered[pid], ids)) for pid in range(self.n_preds)]

        self.filter_bouts(filters=pred_bout_ids_filters)

    def organize_bout_info(self):
        pred_bout_bounds_filtered = list(itertools.chain.from_iterable(self.pred_bout_bounds_filtered))
        self.pred_data_arrs_bouts = [np.vstack([pred_data_arr[pred_bout_bound_filtered[0]:pred_bout_bound_filtered[1]] for pred_bout_bound_filtered in pred_bout_bounds_filtered]) for pred_data_arr in self.pred_data_arrs]
        self.agent_data_arrs_bouts = [np.vstack(
            [agent_data_arr[pred_bout_bound_filtered[0]:pred_bout_bound_filtered[1]] for pred_bout_bound_filtered in pred_bout_bounds_filtered]) for agent_data_arr in self.agent_data_arrs]
        self.timestamps_bouts = np.concatenate(
            [self.timestamps[pred_bout_bound_filtered[0]:pred_bout_bound_filtered[1]] for pred_bout_bound_filtered in pred_bout_bounds_filtered])
        self.agent_com_bouts = np.concatenate([self.agent_com[pred_bout_bound_filtered[0]:pred_bout_bound_filtered[1]] for pred_bout_bound_filtered in pred_bout_bounds_filtered])

        # discarded bouts
        pred_bout_bounds_discarded = [list(itertools.chain.from_iterable(self.pred_bout_bounds_discarded[i])) for i in range(len(self.pred_bout_bounds_discarded))]

        self.pred_data_arrs_bouts_discarded = [[np.vstack(
            [pred_data_arr[pred_bout_bound_discarded[0]:pred_bout_bound_discarded[1]] for pred_bout_bound_discarded in
             pred_bout_bounds_discarded[i]]) for pred_data_arr in self.pred_data_arrs] if len(pred_bout_bounds_discarded[i]) > 0 else [] for i in range(len(pred_bout_bounds_discarded))]
        self.agent_data_arrs_bouts_discarded = [[np.vstack(
            [agent_data_arr[pred_bout_bound_discarded[0]:pred_bout_bound_discarded[1]] for pred_bout_bound_discarded in
             pred_bout_bounds_discarded[i]]) for agent_data_arr in self.agent_data_arrs] if len(pred_bout_bounds_discarded[i]) > 0 else [] for i in range(len(pred_bout_bounds_discarded))]
        self.timestamps_bouts_discarded = [np.concatenate(
            [self.timestamps[pred_bout_bound_discarded[0]:pred_bout_bound_discarded[1]] for pred_bout_bound_discarded in
             pred_bout_bounds_discarded[i]]) if len(pred_bout_bounds_discarded[i]) > 0 else [] for i in range(len(pred_bout_bounds_discarded))]
        self.agent_com_bouts_discarded = [np.concatenate(
            [self.agent_com[pred_bout_bound_discarded[0]:pred_bout_bound_discarded[1]] for pred_bout_bound_discarded in
             pred_bout_bounds_discarded[i]]) if len(pred_bout_bounds_discarded[i]) > 0 else [] for i in range(len(pred_bout_bounds_discarded))]

    def compute_bout_evasion_bounds(self,
                                    vel_angle_change_limit: float = 90.,
                                    pred_dist_to_com_limit: float = 0.4,
                                    jitter_reset_mechanism: bool = True,
                                    check_prey_on_both_sides: bool = True,
                                    margin: int = 0):
        print("Computing bout evasion bounds")
        n_preys_behind_pred = self.compute_n_preys_behind_predator()
        preys_behind_pred = self.compute_preys_behind_predator()
        pred_vel_arrs = self.compute_predator_velocity()
        pred_dist_to_com_arrs = self.compute_predator_distance_to_agent_com()
        pred_prey_sides = self.check_which_side_of_predator_preys_on()

        self.bout_evasion_start_times = [[] for _ in range(self.n_preds)]
        self.bout_evasion_start_ids = [[] for _ in range(self.n_preds)]

        self.bout_evasion_end_times = [[] for _ in range(self.n_preds)]
        self.bout_evasion_end_ids = [[] for _ in range(self.n_preds)]
        for pid in range(self.n_preds):
            evasions_count = 0
            for bout_start, bout_end in self.pred_bout_bounds_filtered[pid]:
                n_preys_behind_pred_bout = n_preys_behind_pred[pid][bout_start:bout_end]
                timestamps_bout = self.timestamps[bout_start:bout_end]
                pred_dist_to_com_bout = pred_dist_to_com_arrs[pid][bout_start:bout_end]
                pred_vel_bout = pred_vel_arrs[pid][bout_start:bout_end]
                pred_prey_sides_bout = pred_prey_sides[pid][bout_start:bout_end]
                preys_behind_pred_bout = preys_behind_pred[pid][bout_start:bout_end]
                all_preys_in_front_prev = False
                prey_behind_on_both_sides_during_evasion = False
                evasion_started = False
                evasion_ended = False
                for tid in range(len(timestamps_bout)):
                    n_preys_behind = n_preys_behind_pred_bout[tid]

                    dist_to_com_condition = True if pred_dist_to_com_limit is None else pred_dist_to_com_bout[tid] <= pred_dist_to_com_limit*self.width/2

                    pred_vel_change = compute_angle(pred_vel_bout[tid-1], np.array([0., 0.]), pred_vel_bout[tid])
                    vel_angle_change_condition = True if vel_angle_change_limit is None else pred_vel_change <= vel_angle_change_limit

                    conditions = not tid == 0 and dist_to_com_condition and vel_angle_change_condition

                    prey_behind_sides = pred_prey_sides_bout[tid][preys_behind_pred_bout[tid] == 1] if check_prey_on_both_sides else True
                    prey_behind_from_both_sides = np.any(prey_behind_sides == 1) and np.any(prey_behind_sides == -1)

                    if conditions:
                        if n_preys_behind == 0:
                            all_preys_in_front_prev = True

                        if not evasion_started:
                            if all_preys_in_front_prev and self.n_agents > n_preys_behind > 0 and tid != len(n_preys_behind_pred_bout) - 1:
                                start_idx = max([0, tid - margin])
                                self.bout_evasion_start_times[pid].append(timestamps_bout[start_idx])
                                self.bout_evasion_start_ids[pid].append(start_idx)
                                evasions_count += 1
                                evasion_started = True
                                all_preys_in_front_prev = False
                        else:
                            if jitter_reset_mechanism and not evasion_ended and n_preys_behind == 0:
                                self.bout_evasion_start_times[pid].pop()
                                self.bout_evasion_start_ids[pid].pop()
                                evasions_count -= 1
                                evasion_started = False
                                prey_behind_on_both_sides_during_evasion = False
                            elif prey_behind_from_both_sides:
                                prey_behind_on_both_sides_during_evasion = True

                            elif n_preys_behind == self.n_agents:
                                if prey_behind_on_both_sides_during_evasion:
                                    end_idx = min([len(timestamps_bout) - 1, tid + margin])
                                    self.bout_evasion_end_times[pid].append(timestamps_bout[end_idx])
                                    self.bout_evasion_end_ids[pid].append(end_idx)
                                    evasion_ended = True

                                break

                if evasion_started and not evasion_ended:
                    if self.bout_evasion_start_ids[pid][-1] == len(timestamps_bout) - 1:
                        print("evasion started at end of bout")
                        self.bout_evasion_start_times[pid].pop()
                        self.bout_evasion_start_ids[pid].pop()
                        evasions_count -= 1
                        evasion_started = False
                    elif not prey_behind_on_both_sides_during_evasion:
                        self.bout_evasion_start_times[pid].pop()
                        self.bout_evasion_start_ids[pid].pop()
                        evasions_count -= 1
                        evasion_started = False
                    else:
                        end_idx = len(timestamps_bout) - 1
                        self.bout_evasion_end_times[pid].append(timestamps_bout[end_idx])
                        self.bout_evasion_end_ids[pid].append(end_idx)
                if not evasion_started:
                    self.bout_evasion_start_times[pid].append(-1)
                    self.bout_evasion_start_ids[pid].append(-1)
                    self.bout_evasion_end_times[pid].append(-1)
                    self.bout_evasion_end_ids[pid].append(-1)

            print(f"--> {evasions_count} evasions detected for predator {pid + 1}")

    def compute_bout_evasion_straightness_metric(self, margin: int = 0) -> List[NDArray[float]]:
        preds_bout_evasion_straightness_metric = []
        for pid in range(self.n_preds):
            bout_evasion_straightness_metric = np.full(len(self.pred_bout_bounds_filtered[pid]), -1.)
            for bout_id in range(len(self.pred_bout_bounds_filtered[pid])):
                if self.bout_evasion_start_ids[pid][bout_id] >= 0:
                    bout_start, _ = self.pred_bout_bounds_filtered[pid][bout_id]
                    evasion_start, evasion_end = self.bout_evasion_start_ids[pid][bout_id], self.bout_evasion_end_ids[pid][bout_id]
                    evasion_positions = self.pred_data_arrs[pid][max([0, bout_start + evasion_start-margin]):min([bout_start + evasion_end+margin, len(self.pred_data_arrs[pid])])]

                    distance_traveled = compute_distance_traveled(evasion_positions)
                    displacement = euclidean_distance(evasion_positions[0], evasion_positions[-1])
                    bout_evasion_straightness_metric[bout_id] = displacement/distance_traveled

            preds_bout_evasion_straightness_metric.append(bout_evasion_straightness_metric)

        return preds_bout_evasion_straightness_metric

    def compute_bout_evasion_fountain_metric(self, margin: int = 0, method: str = "distance", n_closest_agents: int = 10) -> List[NDArray[float]]:
        agents_vel = self.compute_agent_velocity()
        preds_vel = self.compute_predator_velocity()
        preds_dist_to_agents = self.compute_predator_distance_to_agents()

        preds_bout_evasion_fountain_metric = []
        for pid in range(self.n_preds):
            bout_evasion_fountain_metric = np.full(len(self.pred_bout_bounds_filtered[pid]), -1.)
            for bout_id in range(len(self.pred_bout_bounds_filtered[pid])):
                if self.bout_evasion_start_ids[pid][bout_id] >= 0:
                    bout_start, _ = self.pred_bout_bounds_filtered[pid][bout_id]
                    evasion_start = max([0, bout_start + self.bout_evasion_start_ids[pid][bout_id]-margin])
                    evasion_end = min([bout_start + self.bout_evasion_end_ids[pid][bout_id]+margin, len(self.pred_data_arrs[pid])])

                    pred_evasion_positions = self.pred_data_arrs[pid][evasion_start:evasion_end]
                    agents_evasion_positions = turn_list_of_2d_arrays_into_3d_array(transpose_list_of_arrays([self.agent_data_arrs[aid][evasion_start:evasion_end] for aid in range(self.n_agents)]))

                    if method == "distance":
                        pred_evasion_vel = preds_vel[pid][evasion_start:evasion_end]
                        agents_evasion_vel = turn_list_of_2d_arrays_into_3d_array(transpose_list_of_arrays([agents_vel[aid][evasion_start:evasion_end] for aid in range(self.n_agents)]))

                        pred_evasion_dist_to_agents = preds_dist_to_agents[pid][evasion_start:evasion_end]
                        flee_arr = np.full_like(pred_evasion_dist_to_agents, 0)
                        for tid in range(flee_arr.shape[0]):
                            n_closest_agents_ids = np.argpartition(pred_evasion_dist_to_agents[tid], n_closest_agents)[:n_closest_agents]
                            flee_arr[tid, n_closest_agents_ids] = 50

                        bout_evasion_fountain_metric[bout_id] = np.mean(extract_SII(pos_rep=agents_evasion_positions, vel_rep=agents_evasion_vel,
                                                                                    pred_pos_rep=pred_evasion_positions, pred_vel_rep=pred_evasion_vel,
                                                                                    flee=flee_arr))
            preds_bout_evasion_fountain_metric.append(bout_evasion_fountain_metric)

        return preds_bout_evasion_fountain_metric

    def compute_bout_evasion_circularity(self, margin: int = 0, all_timepoints: bool = False) -> Union[List[NDArray[float]], List[List[NDArray[float]]]]:
        preds_dist_to_agent_com = self.compute_predator_distance_to_agent_com()

        preds_bout_evasion_circularity_metric = []
        self.preds_convex_hulls = []
        self.preds_convex_hull_points = []
        self.preds_bounding_circles = []
        for pid in range(self.n_preds):
            bout_evasion_circularity_metric = np.full(len(self.pred_bout_bounds_filtered[pid]), -1.) if not all_timepoints else []
            convex_hulls = []
            convex_hull_points = []
            bounding_circles = []
            for bout_id in range(len(self.pred_bout_bounds_filtered[pid])):
                if all_timepoints or self.bout_evasion_start_ids[pid][bout_id] >= 0:
                    bout_start, bout_end = self.pred_bout_bounds_filtered[pid][bout_id]
                    evasion_start = max([0, bout_start + self.bout_evasion_start_ids[pid][bout_id] - margin])
                    evasion_end = min([bout_start + self.bout_evasion_end_ids[pid][bout_id] + margin, len(self.pred_data_arrs[pid])])

                    agents_evasion_positions = turn_list_of_2d_arrays_into_3d_array(transpose_list_of_arrays([self.agent_data_arrs[aid][bout_start:bout_end] for aid in range(self.n_agents)]))

                    if all_timepoints:
                        ids = range(agents_evasion_positions.shape[0])
                    else:
                        pred_dist_to_agent_com = preds_dist_to_agent_com[pid][evasion_start:evasion_end]

                        min_pred_dist_to_com_idx = np.argmin(pred_dist_to_agent_com) + max([0, self.bout_evasion_start_ids[pid][bout_id] - margin])
                        ids = [min_pred_dist_to_com_idx]

                    bout_evasion_circularity_timepoints = []
                    for id in ids:
                        agent_pos_at_min_pred_dist_to_com = agents_evasion_positions[id]

                        convexhull = scipy.spatial.ConvexHull(agent_pos_at_min_pred_dist_to_com)
                        convexhull_points = agent_pos_at_min_pred_dist_to_com[convexhull.vertices]
                        convexhull_polygon = shapely.geometry.Polygon(convexhull_points)

                        min_bounding_circle = shapely.minimum_bounding_circle(convexhull_polygon)
                        circularity = convexhull_polygon.area / min_bounding_circle.area

                        if not all_timepoints:
                            bout_evasion_circularity_metric[bout_id] = circularity
                            bounding_circles.append(min_bounding_circle.exterior.coords)
                            convex_hulls.append(convexhull)
                            convex_hull_points.append(agent_pos_at_min_pred_dist_to_com)
                        else:
                            bout_evasion_circularity_timepoints.append(circularity)

                    if all_timepoints:
                        bout_evasion_circularity_metric.append(np.array(bout_evasion_circularity_timepoints))

                else:
                    bounding_circles.append(None)
                    convex_hulls.append(None)
                    convex_hull_points.append(None)

            self.preds_bounding_circles.append(bounding_circles)
            self.preds_convex_hull_points.append(convex_hull_points)
            self.preds_convex_hulls.append(convex_hulls)
            preds_bout_evasion_circularity_metric.append(bout_evasion_circularity_metric)

        return preds_bout_evasion_circularity_metric

    def compute_bout_evasion_convexity(self, margin: int = 0, compute_at: str = "end") -> Union[List[NDArray[float]], List[List[NDArray[float]]]]:
        preds_bout_evasion_convexity_metric = []
        for pid in range(self.n_preds):
            bout_evasion_convexity_metric = np.full(len(self.pred_bout_bounds_filtered[pid]), -1.) if compute_at != "all" else []
            for bout_id in range(len(self.pred_bout_bounds_filtered[pid])):
                if compute_at == "all" or self.bout_evasion_start_ids[pid][bout_id] >= 0:
                    bout_start, bout_end = self.pred_bout_bounds_filtered[pid][bout_id]

                    if compute_at == "end":
                        evasion_end = min([bout_start + self.bout_evasion_end_ids[pid][bout_id] + margin, len(self.pred_data_arrs[pid])])
                        agents_evasion_positions = [np.vstack([self.agent_data_arrs[aid][evasion_end] for aid in range(self.n_agents)])]
                    elif compute_at == "middle":
                        evasion_middle = int(bout_start + (self.bout_evasion_start_ids[pid][bout_id] + self.bout_evasion_end_ids[pid][bout_id]) / 2)
                        agents_evasion_positions = [np.vstack([self.agent_data_arrs[aid][evasion_middle] for aid in range(self.n_agents)])]
                    elif compute_at == "all":
                        agents_evasion_positions = transpose_list_of_arrays([self.agent_data_arrs[aid][bout_start:bout_end] for aid in range(self.n_agents)])
                    else:
                        raise ValueError(f"Invalid compute_at value: {compute_at}")

                    bout_evasion_convexity_metric_timepoints = []
                    for timepoint in agents_evasion_positions:
                        shaper = alpha_shapes.Alpha_Shaper(timepoint)
                        alpha_opt, alpha_shape = shaper.optimize()

                        if "Multi" in str(type(alpha_shape.boundary)):
                            alpha_coords = []
                            for geom in alpha_shape.boundary.geoms:
                                alpha_x, alpha_y = geom.coords.xy
                                if not geom.is_closed:
                                    alpha_x = np.array(list(alpha_x) + [alpha_x[0]])
                                    alpha_y = np.array(list(alpha_y) + [alpha_y[0]])
                                alpha_coords.append(np.vstack([alpha_x, alpha_y]).T)
                            alpha_coords = np.vstack(alpha_coords)
                        else:
                            alpha_x, alpha_y = alpha_shape.boundary.coords.xy
                            if not alpha_shape.is_closed:
                                alpha_x = np.array(list(alpha_x) + [alpha_x[0]])
                                alpha_y = np.array(list(alpha_y) + [alpha_y[0]])
                            alpha_coords = np.vstack([alpha_x, alpha_y]).T

                        convexhull = scipy.spatial.ConvexHull(alpha_coords)
                        convexhull_coords = alpha_coords[convexhull.vertices]
                        convexity = shapely.Polygon(convexhull_coords).length / shapely.Polygon(alpha_coords).length
                        if compute_at != 'all':
                            bout_evasion_convexity_metric[bout_id] = convexity
                        else:
                            bout_evasion_convexity_metric_timepoints.append(convexity)
                    if compute_at == "all":
                        bout_evasion_convexity_metric.append(np.array(bout_evasion_convexity_metric_timepoints))

            preds_bout_evasion_convexity_metric.append(bout_evasion_convexity_metric)

        return preds_bout_evasion_convexity_metric

    def compute_bout_evasion_polarisation(self, margin: int = 0, compute_at: str = "end") -> Union[List[NDArray[float]], List[List[NDArray[float]]]]:
        agents_polarisation = self.compute_agent_polarisation()

        preds_bout_evasion_polarisation_metric = []
        for pid in range(self.n_preds):
            bout_evasion_polarisation_metric = np.full(len(self.pred_bout_bounds_filtered[pid]), -1.) if compute_at != "all" else []
            for bout_id in range(len(self.pred_bout_bounds_filtered[pid])):
                if compute_at == "all" or self.bout_evasion_start_ids[pid][bout_id] >= 0:
                    bout_start, bout_end = self.pred_bout_bounds_filtered[pid][bout_id]

                    if compute_at == "end":
                        evasion_end = min([bout_start + self.bout_evasion_end_ids[pid][bout_id] + margin, len(self.pred_data_arrs[pid])])
                        polarisation = agents_polarisation[evasion_end]
                    elif compute_at == "middle":
                        evasion_middle = int(bout_start + (self.bout_evasion_start_ids[pid][bout_id] + self.bout_evasion_end_ids[pid][bout_id]) / 2)
                        polarisation = agents_polarisation[evasion_middle]
                    elif compute_at == "all":
                        polarisation = agents_polarisation[bout_start:bout_end]
                    else:
                        raise ValueError(f"Invalid compute_at value: {compute_at}")

                    if compute_at != "all":
                        bout_evasion_polarisation_metric[bout_id] = polarisation
                    else:
                        bout_evasion_polarisation_metric.append(polarisation)

            preds_bout_evasion_polarisation_metric.append(bout_evasion_polarisation_metric)

        return preds_bout_evasion_polarisation_metric

    def detect_isolated_agents(self, dbscan_eps: float = 0.15) -> List[List[List[List[int]]]]:
        agent_dist_to_nn = transpose_list_of_arrays(self.compute_agent_distance_to_nearest_neighbor())

        isolated_individuals = []
        for pid in range(self.n_preds):
            pred_isolated_individuals = []
            for bout_id in range(len(self.pred_bout_bounds_filtered[pid])):
                bout_start, bout_end = self.pred_bout_bounds_filtered[pid][bout_id]
                pred_bout_isolated_individuals = []
                for tid in range(bout_start, bout_end):
                    data = agent_dist_to_nn[tid]
                    dbscan = sklearn.cluster.DBSCAN(eps=dbscan_eps)
                    outliers = dbscan.fit_predict(data)
                    pred_bout_isolated_individuals.append([outlier[0] for outlier in np.argwhere(outliers != 0)])
                pred_isolated_individuals.append(pred_bout_isolated_individuals)
            isolated_individuals.append(pred_isolated_individuals)

        return isolated_individuals

    def write_bout_info(self, output_path, exp_plotter_args: Dict[str, Any],
                        mark_speed_spike: bool = False, speed_spike_threshold: float = 10.) -> None:
        exp_plotter = RoundPlotter(self, **exp_plotter_args)
        com_speed = self.compute_agent_com_speed()
        preds_speed = self.compute_predator_speed()
        preds_acc = self.compute_predator_acceleration(smooth=False)
        preds_attack_angle = self.compute_predator_attack_angle(smooth=False)
        preds_n_preys_behind = self.compute_n_preys_behind_predator()

        circularity = self.compute_bout_evasion_circularity(all_timepoints=True)
        convexity = self.compute_bout_evasion_convexity(compute_at="all")
        polarisation = self.compute_bout_evasion_polarisation(compute_at="all")

        for pid_in_bout in range(self.n_preds):
            for bout_id in range(len(self.pred_bout_bounds_filtered[pid_in_bout])):
                print(f"Writing bout {self.pred_bout_ids_filtered[pid_in_bout][bout_id]}...")
                
                bout_start, bout_end = self.pred_bout_bounds_filtered[pid_in_bout][bout_id]

                bout_output_path = f"{output_path}/{self.pred_bout_ids_filtered[pid_in_bout][bout_id]}"
                if not os.path.exists(bout_output_path):
                    os.makedirs(bout_output_path)

                # CSV file
                output_csv = f"{bout_output_path}/{self.pred_bout_ids_filtered[pid_in_bout][bout_id]}.csv"

                if os.path.isfile(output_csv):  # Empty the file if it already exists
                    f = open(output_csv, "w+")
                    f.close()

                with open(output_csv, 'a') as output_file:
                    header = ",".join(["timestep", "timestamp", "coms", "circularity", "convexity", "polarisation"])
                    for aid in range(self.n_agents):
                        header = ",".join([header, f"x{aid}", f"y{aid}"])
                    for pid in range(self.n_preds):
                        header = ",".join([header, f"prx{pid}", f"pry{pid}",
                                                   f"prs{pid}", f"pra{pid}", f"praa{pid}", f"prnpb{pid}"])
                    output_file.write(f"{header}\n")

                    for idx in range(bout_start, bout_end):
                        line = ",".join([str(self.timesteps[idx]), str(self.timestamps[idx]), str(com_speed[idx]),
                                         str(circularity[pid_in_bout][bout_id][idx - bout_start]),
                                         str(convexity[pid_in_bout][bout_id][idx - bout_start]),
                                         str(polarisation[pid_in_bout][bout_id][idx - bout_start])])
                        for aid in range(self.n_agents):
                            line = ",".join([line, str(self.agents_data[aid][idx][0]), str(self.agents_data[aid][idx][1])])
                        for pid in range(self.n_preds):
                            line = ",".join([line, str(self.preds_data[pid][idx][0]), str(self.preds_data[pid][idx][1]),
                                                   str(preds_speed[pid][idx]), str(preds_acc[pid][idx]),
                                                   str(preds_attack_angle[pid][idx]), str(preds_n_preys_behind[pid][idx])])
                        output_file.write(f"{line}\n")

                # Video
                # change to single bout
                exp_plotter.plot_single_bout(bout_id=self.pred_bout_ids_filtered[pid_in_bout][bout_id], show_com=True, save=True, show=False,
                                             mark_speed_spike=mark_speed_spike, speed_spike_threshold=speed_spike_threshold,
                                             out_file_path=f"{bout_output_path}/bout_divisions_{self.pred_bout_ids_filtered[pid_in_bout][bout_id]}.mp4")


if __name__ == "__main__":
    #find CoBeHumanExperimentsData/ -name '*.zip' -exec sh -c 'unzip -d "${1%.*}" "$1"' _ {} \;

    round_id = "6095799011"#"2431419351"
    round_types = ["P1", "P2", "Shared", "P1R1", "P1R2", "P1R3"]
    round_type_id = 2
    file_path = f"./CoBeHumanExperimentsData/{round_id}/{round_types[round_type_id-1] if round_type_id < 4 else round_types[round_type_id-1][:2].upper()}/{round_id}_{round_types[round_type_id-1][:2].upper() if round_type_id < 4 else round_types[round_type_id-1]}.csv"

    if not os.path.exists(file_path):
        file_path = f"./CoBeHumanExperimentsData/{round_id}/{round_types[round_type_id-1][:2].upper()}/{round_id}_{round_types[round_type_id-1][:2].upper()}.csv"
        round_types[round_type_id - 1] = round_types[round_type_id -1][:2].upper()

    exp = Round(file_path, n_preds=2 if round_type_id == 3 else 1,
    if not os.path.exists(file_path):
        file_path = f"./CoBeHumanExperimentsData/{round_id}/{round_types[round_type_id-1][:2].upper()}/{round_id}_{round_types[round_type_id-1][:2]}_{round_types[round_type_id - 1][2:]}.csv"
                dist_tolerance=0.7, margin=10, min_bout_length=30,
                speed_tolerance=0.15, speed_threshold=0.475, absolute_speed_threshold=6.65, #0.15, 5., 70.
                bout_ids_to_remove=None, evasion_pred_dist_to_com_limit=None, evasion_vel_angle_change_limit=None)

    exp_plotter = RoundPlotter(exp, mac=True)
    exp_plotter.plot_metrics(time_window_dur=2, smoothing_args={'kernel': gaussian, 'window_size': 100},
                             #save=True, # saving does not seem to work yet
                             out_file_path=f"./CoBeHumanExperimentsDataAnonymized/{round_id}/{round_types[round_type_id-1]}/{round_id}_{round_types[round_type_id-1][:2].upper()}.mp4",
                             com_only=True)
    exp_plotter.plot_predator_acc_smoothings(time_window_dur=2, window_size=40, com_only=True)
    exp_plotter.plot_bout_trajectories()
    exp_plotter.plot_bout_trajectories(discarded=True, filter_number=0)  # minimum bout length filter
    exp_plotter.plot_bout_trajectories(discarded=True, filter_number=1)  # speed filter
    exp_plotter.plot_bout_trajectories(discarded=True, filter_number=2)  # absolute speed filter
    exp_plotter.plot_bout_division(com_only=True)
