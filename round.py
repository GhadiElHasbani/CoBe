import itertools
from tinyflux import TinyFlux
from helpers import *
from round_plotter import RoundPlotter
from typing import Dict
import os
import file_last_modification_time_finder


class Round:

    def __init__(self,
                 file_path: str,
                 T_start: int = 0,
                 T: int = None,
                 n_agents: int = 50,
                 n_preds: int = 1,
                 center: Tuple[float, float] = (0, 0),
                 radius: float = 20,
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

        self.center = center
        self.radius = radius
        self.width = ((2 ** 0.5) * self.radius + self.radius * 2) / 2
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
            agent_data = [(agent_x[i] + self.center[0], agent_y[i] + self.center[1]) for i in range(len(timesteps))]
            # sorting agent data according to timesteps list
            agent_data = [agent_data[i] for i in sorted(range(len(timesteps)), key=lambda x: timesteps[x])]
            # slicing to length
            agent_data = agent_data[self.T_start:self.T]
            agent_data_arr = convert_list_of_tuples_to_array(agent_data)
            self.agents_data.append(agent_data)
            self.agent_data_arrs.append(agent_data_arr)

        # retrieving data for predator agent(s)
        self.pred_datas = []
        self.pred_data_arrs = []

        for pred_id in range(self.n_preds):
            pred_x = self.db.get_field_values("prx" + str(pred_id))
            pred_y = self.db.get_field_values("pry" + str(pred_id))
            pred_data = [(pred_x[i] + self.center[0], pred_y[i] + self.center[1]) for i in range(len(timesteps))]
            # sorting pred data according to timesteps list
            pred_data = [pred_data[i] for i in sorted(range(len(timesteps)), key=lambda x: timesteps[x])]
            # slicing to length
            pred_data = pred_data[self.T_start:self.T]
            pred_data_arr = convert_list_of_tuples_to_array(pred_data)
            self.pred_datas.append(pred_data)
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
            self.pred_datas = [np.array(pred_data)[dup_filter] for pred_data in self.pred_datas]
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
        self.pred_bout_ids = None

        self.pred_bout_bounds_filtered = None
        self.pred_bout_bounds_discarded = []
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
        json_files = file_last_modification_time_finder.get_json_files(json_file_folder)
        print(f"Found {len(json_files)} json files in the folder.")

        # ordering files by filename
        json_files = file_last_modification_time_finder.sort_files_by_filename(json_files)

        json_files = [os.path.join(json_file_folder, file) for file in json_files]
        print(
            f"Files last modified between {file_last_modification_time_finder.file_last_modification_time_minutes_seconds(json_files[0])} and {file_last_modification_time_finder.file_last_modification_time_minutes_seconds(json_files[-1])}.")

        # getting the number of files that were in a given second
        _, num_files_between = file_last_modification_time_finder.find_consecutive_files_with_increasing_seconds(json_files)

        # calculated framerate in each minute
        framerates_per_min = [rate for rate in num_files_between]

        # removing first and last elements as these can be not full minutes
        framerates_per_min = framerates_per_min[1:-1]

        avg_fs = sum(framerates_per_min) / len(framerates_per_min)
        return avg_fs

    def compute_speed(self, data_arrs: List[NDArray[float]]) -> List[NDArray[float]]:
        return [
            np.insert(np.sqrt(np.sum(np.diff(data_arr, axis=0) ** 2, axis=1)) / (np.diff(self.timestamps)), 0, [0.,]) for
            data_arr in data_arrs]

    def compute_velocity(self, data_arrs: List[NDArray[float]]) -> List[NDArray[float]]:
        return [np.vstack([np.diff(data_arr, axis=0) / (np.diff(self.timestamps).reshape((-1, 1))), [0., 0.]]) for data_arr in data_arrs]

    def compute_predator_velocity(self):
        return self.compute_velocity(self.pred_data_arrs)

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
        pred_datas_speed = self.compute_predator_speed()
        pred_datas_acc = self.compute_acceleration(pred_datas_speed, smooth, smoothing_args)
        return pred_datas_acc

    def compute_predator_distance_to_agent_com(self) -> List[NDArray[float]]:
        pred_dist_to_agent_com = [np.sqrt(np.sum((pred_data_arr - self.agent_com)**2, axis = 1)) for pred_data_arr in self.pred_data_arrs]

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
                    agent_pos_vec = np.array(self.agents_data[aid][tid]) - self.pred_datas[pid][tid]
                    preys_behind[pid][tid, aid] = 1 if agent_pos_vec.reshape((1, -1)) @ pred_vel_vec.reshape((-1, 1)) < 0 else 0

        return preys_behind

    def compute_n_preys_behind_predator(self):
        preys_behind = self.compute_preys_behind_predator()

        return [np.sum(preys_behind[pid], axis=-1) for pid in range(self.n_preds)]

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
        self.pred_bout_ids = [np.arange(len(self.pred_bout_bounds[pid])).astype(str) + np.full(len(self.pred_bout_bounds[pid]), '_' + str(pid)) for pid in range(self.n_preds)]
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
        self.pred_bout_bounds_discarded.append(
            [np.array(self.pred_bout_bounds_filtered[pid])[np.logical_not(filters[pid])] for pid in
             range(self.n_preds)])
        self.pred_bout_bounds_filtered = [
            np.array(self.pred_bout_bounds_filtered[pid])[filters[pid]] for pid in
            range(self.n_preds)]
        self.pred_bout_ids = [self.pred_bout_ids[pid][filters[pid]] for pid in range(self.n_preds)]

    def apply_bout_length_filter(self, min_bout_length: int):
        print(f"Discarding bouts with less than {min_bout_length} observations")

        pred_bout_length_filters = [np.array(self.pred_bout_lengths[pid]) >= min_bout_length for pid in
                                    range(self.n_preds)]

        self.filter_bouts(filters=pred_bout_length_filters)

        print(f"--> {np.sum([len(self.pred_bout_bounds_filtered[pid]) for pid in range(self.n_preds)])} remain")

    def apply_bout_low_speed_filter(self, speed_threshold: float, speed_tolerance: float):
        print(f"Discarding bouts with speed less than {speed_threshold} for more than {speed_tolerance*100}% of observations")
        pred_datas_vel = self.compute_predator_speed()
        pred_bouts_vel = [[pred_datas_vel[pid][self.pred_bout_bounds_filtered[pid][i][0]:self.pred_bout_bounds_filtered[pid][i][1]] for i in range(len(self.pred_bout_bounds_filtered[pid]))] for pid in range(self.n_preds)]
        pred_bout_speed_filters = [np.array([not np.sum(np.array(pred_bouts_vel[pid][i]) < speed_threshold)/len(pred_bouts_vel[pid][i]) > speed_tolerance for i in range(len(pred_bouts_vel[pid]))]) for pid in range(self.n_preds)]

        self.filter_bouts(filters=pred_bout_speed_filters)

        print(f"--> {np.sum([len(self.pred_bout_bounds_filtered[pid]) for pid in range(self.n_preds)])} remain")

    def apply_bout_high_speed_filter(self, absolute_speed_threshold: float):
        print(f"Discarding bouts with speed greater than {absolute_speed_threshold}")

        pred_datas_vel = self.compute_predator_speed()
        pred_dists_to_com = self.compute_predator_distance_to_agent_com()
        pred_dists_to_border = self.compute_predator_distance_to_border()
        pred_bout_abs_speed_filters = [[] for _ in range(self.n_preds)]
        for pid in range(self.n_preds):
            for pred_bout_bound in self.pred_bout_bounds_filtered[pid]:
                pred_bouts_vel = [pred_datas_vel[pid][pred_bout_bound[0]:pred_bout_bound[1]] for pid in
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
        pred_bout_ids_filters = [np.logical_not(np.isin(self.pred_bout_ids[pid], ids)) for pid in range(self.n_preds)]

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
                                    pred_dist_to_com_limit: float = 0.4):
        print("Computing bout evasion start times")
        n_preys_behind_pred = self.compute_n_preys_behind_predator()
        pred_vel_arrs = self.compute_predator_velocity()
        pred_dist_to_com_arrs = self.compute_predator_distance_to_agent_com()

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
                all_preys_in_front_prev = False
                evasion_started = False
                evasion_ended = False
                for i in range(len(n_preys_behind_pred_bout)):
                    n_preys_behind = n_preys_behind_pred_bout[i]
                    pred_vel_change = compute_angle(pred_vel_bout[i-1], np.array([0., 0.]), pred_vel_bout[i])
                    dist_to_com_condition = True if pred_dist_to_com_limit is None else pred_dist_to_com_bout[i] <= pred_dist_to_com_limit*self.width/2
                    vel_angle_change_condition = True if vel_angle_change_limit is None else pred_vel_change <= vel_angle_change_limit
                    conditions = True if pred_dist_to_com_limit is None and vel_angle_change_limit is None else not i == 0 and dist_to_com_condition and vel_angle_change_condition
                    if conditions:
                        if n_preys_behind == 0:
                            all_preys_in_front_prev = True

                        if all_preys_in_front_prev and n_preys_behind > 0 and not evasion_started:
                            self.bout_evasion_start_times[pid].append(timestamps_bout[i])
                            self.bout_evasion_start_ids[pid].append(i)
                            evasions_count += 1
                            evasion_started = True
                        elif evasion_started and n_preys_behind == self.n_agents:
                            self.bout_evasion_end_times[pid].append(timestamps_bout[i])
                            self.bout_evasion_end_ids[pid].append(i)
                            evasion_ended = True
                            break
                if evasion_started and not evasion_ended:
                    self.bout_evasion_end_times[pid].append(timestamps_bout[i])
                    self.bout_evasion_end_ids[pid].append(i)
                elif not evasion_started:
                    self.bout_evasion_start_times[pid].append(-1)
                    self.bout_evasion_start_ids[pid].append(-1)
                    self.bout_evasion_end_times[pid].append(-1)
                    self.bout_evasion_end_ids[pid].append(-1)

            print(f"--> {evasions_count} evasions detected for predator {pid + 1}")


if __name__ == "__main__":
    #find CoBeHumanExperimentsData/ -name '*.zip' -exec sh -c 'unzip -d "${1%.*}" "$1"' _ {} \;

    round_id = "2837465091"
    round_types = ["P1", "P2", "Shared", "P1R1", "P1R2", "P1R3"]
    round_type_id = 2
    file_path = f"./CoBeHumanExperimentsData/{round_id}/{round_types[round_type_id-1] if round_type_id < 4 else round_types[round_type_id-1][:2].upper()}/{round_id}_{round_types[round_type_id-1][:2].upper() if round_type_id < 4 else round_types[round_type_id-1]}.csv"

    if not os.path.exists(file_path):
        file_path = f"./CoBeHumanExperimentsData/{round_id}/{round_types[round_type_id-1][:2].upper()}/{round_id}_{round_types[round_type_id-1][:2].upper()}.csv"
        round_types[round_type_id - 1] = round_types[round_type_id -1][:2].upper()

    exp = Round(file_path, n_preds=2 if round_type_id == 3 else 1,
                dist_tolerance=0.7, margin=10, min_bout_length=30,
                speed_tolerance=0.15, speed_threshold=5., absolute_speed_threshold=70,
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
