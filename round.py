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
                 speed_tolerance: float = 1.):
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
        self.agent_datas = []
        self.agent_datas_arr = []

        for agent_id in range(self.n_agents):
            agent_x = self.db.get_field_values("x" + str(agent_id))
            agent_y = self.db.get_field_values("y" + str(agent_id))
            agent_data = [(agent_x[i], agent_y[i]) for i in range(len(timesteps))]
            # sorting agent data according to timesteps list
            agent_data = [agent_data[i] for i in sorted(range(len(timesteps)), key=lambda x: timesteps[x])]
            # slicing to length
            agent_data = agent_data[self.T_start:self.T]
            agent_data_arr = convert_list_of_tuples_to_array(agent_data)
            self.agent_datas.append(agent_data)
            self.agent_datas_arr.append(agent_data_arr)

        # retrieving data for predator agent(s)
        self.pred_datas = []
        self.pred_datas_arr = []

        for pred_id in range(self.n_preds):

            pred_x = self.db.get_field_values("prx" + str(pred_id))
            pred_y = self.db.get_field_values("pry" + str(pred_id))
            pred_data = [(pred_x[i], pred_y[i]) for i in range(len(timesteps))]
            # sorting pred data according to timesteps list
            pred_data = [pred_data[i] for i in sorted(range(len(timesteps)), key=lambda x: timesteps[x])]
            # slicing to length
            pred_data = pred_data[self.T_start:self.T]
            pred_data_arr = convert_list_of_tuples_to_array(pred_data)
            self.pred_datas.append(pred_data)
            self.pred_datas_arr.append(pred_data_arr)

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
            self.pred_datas_arr = [np.array(pred_data_arr)[dup_filter] for pred_data_arr in self.pred_datas_arr]
            self.agent_datas = [np.array(agent_data)[dup_filter] for agent_data in self.agent_datas]
            self.agent_datas_arr = [np.array(agent_data_arr)[dup_filter] for agent_data_arr in self.agent_datas_arr]

        self.unit_time = np.array(self.timesteps) / np.array(self.timestamps)

        # storing agent center of mass over time
        self.agent_com = np.mean(np.array(self.agent_datas_arr), axis=0)

        # create bouts
        self.pred_bout_bounds = None
        self.pred_datas_arr_bouts = None
        self.agent_datas_arr_bouts = None
        self.timestamps_bouts = None
        self.agent_com_bouts = None

        self.pred_bout_bounds_discarded = None
        self.pred_datas_arr_bouts_discarded = None
        self.agent_datas_arr_bouts_discarded = None
        self.timestamps_bouts_discarded = None
        self.agent_com_bouts_discarded = None
        self.segment_into_bouts(dist_tolerance=dist_tolerance, margin=margin, min_bout_length=min_bout_length,
                                speed_threshold=speed_threshold, speed_tolerance=speed_tolerance)

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

    def compute_speed(self, datas_arr: List[NDArray]) -> List[NDArray[float]]:
        return [
            np.insert(np.sqrt(np.sum(np.diff(data_arr, axis=0) ** 2, axis=1)) / (np.diff(self.timestamps)), 0, [0.,]) for
            data_arr in datas_arr]

    def compute_agent_speed(self) -> List[NDArray[float]]:
        agent_datas_vel = self.compute_speed(self.agent_datas_arr)
        return agent_datas_vel

    def compute_predator_speed(self) -> List[NDArray[float]]:
        pred_datas_vel = self.compute_speed(self.pred_datas_arr)
        return pred_datas_vel

    def compute_acceleration(self, datas_vel: List[NDArray[float]],
                             smooth: bool = True, smoothing_args: Dict = None) -> List[NDArray[float]]:
        if smooth:
            if smoothing_args is None:
                smoothing_args = {'smoothing_method': 'window',
                                  'kernel': lambda x, i, b: 1,
                                  'window_size': 40}
            datas_acc = [np.insert(smooth_array(np.diff(data_vel) / (np.diff(self.timestamps)), **smoothing_args), 0, [0.,]) for data_vel in datas_vel]
        else:
            datas_acc = [np.insert(np.diff(data_vel) / (np.diff(self.timestamps)), 0, [0.,]) for data_vel in datas_vel]

        return datas_acc

    def compute_agent_acceleration(self, smooth: bool = True, smoothing_args: Dict = None) -> List[NDArray[float]]:
        agent_datas_vel = self.compute_agent_speed()
        agent_datas_acc = self.compute_acceleration(agent_datas_vel, smooth, smoothing_args)

        return agent_datas_acc

    def compute_predator_acceleration(self, smooth: bool = True, smoothing_args: Dict = None) -> List[NDArray[float]]:
        pred_datas_vel = self.compute_predator_speed()
        pred_datas_acc = self.compute_acceleration(pred_datas_vel, smooth, smoothing_args)
        return pred_datas_acc

    def compute_predator_distance_to_agent_com(self) -> List[NDArray[float]]:
        pred_dist_to_agent_com = [np.sqrt(np.sum((pred_data_arr - self.agent_com)**2, axis = 1)) for pred_data_arr in self.pred_datas_arr]

        return pred_dist_to_agent_com

    def compute_predator_distance_to_border(self) -> List[NDArray[float]]: # to implement
        pred_dist_to_border = [shortest_dist_to_polygon(self.arena_points, pred_data) for pred_data in self.pred_datas_arr]

        return pred_dist_to_border

    def compute_agent_com_distance_to_border(self) -> NDArray[float]:  # to implement
        agent_com_dist_to_border = shortest_dist_to_circle(self.center, self.radius, self.agent_com.T)

        return agent_com_dist_to_border

    def compute_predator_distance_to_center(self) -> List[NDArray[float]]:
        preds_dist_to_center = [euclidean_distance(np.array(self.center).reshape((-1,1)), pred_data_arr.T) for pred_data_arr in self.pred_datas_arr]

        return preds_dist_to_center

    def segment_into_bouts(self,
                           dist_tolerance: float = 0.9,
                           margin: int = 0,
                           min_bout_length: int = 2,
                           speed_threshold: float = 0.,
                           speed_tolerance: float = 1.):
        dist_threshold = dist_tolerance*self.width/2

        print(f"Segmenting bouts using threshold of {dist_threshold}, with a margin of {margin} observations, on predator distance from arena center {self.center}")
        pred_dist_to_center = self.compute_predator_distance_to_center()

        pred_points_during_bouts = [pred_dist_to_center < dist_threshold for pred_dist_to_center in pred_dist_to_center]
        temp = [get_bounds(pred_points_during_bouts, margin=margin) for pred_points_during_bouts in pred_points_during_bouts]
        pred_bout_bounds = [temp_el[0] for temp_el in temp]

        print(f"--> found {np.sum([len(pred_bout_bounds[pid]) for pid in range(self.n_preds)])}")

        print(f"Discarding bouts with less than {min_bout_length} observations")
        pred_bout_lengths = [temp_el[1] for temp_el in temp]
        pred_bout_length_filters = [np.array(pred_bout_lengths[pid]) >= min_bout_length for pid in range(self.n_preds)]

        discarded_bouts_length = [np.array(pred_bout_bounds[pid])[np.logical_not(pred_bout_length_filters[pid])].tolist() for pid in range(self.n_preds)]
        pred_bout_bounds_filtered = [np.array(pred_bout_bounds[pid])[pred_bout_length_filters[pid]].tolist() for pid in range(self.n_preds)]

        print(f"--> {np.sum([len(pred_bout_bounds_filtered[pid]) for pid in range(self.n_preds)])} remain")

        print(f"Discarding bouts with speed less than {speed_threshold} for more than {speed_tolerance*100}% of observations")
        pred_datas_vel = self.compute_predator_speed()
        pred_bouts_vel = [[pred_datas_vel[pid][pred_bout_bounds_filtered[pid][i][0]:pred_bout_bounds_filtered[pid][i][1]] for i in range(len(pred_bout_bounds_filtered[pid]))] for pid in range(self.n_preds)]
        pred_bout_speed_filters = [np.array([not np.sum(np.array(pred_bouts_vel[pid][i]) < speed_threshold)/len(pred_bouts_vel[pid][i]) > speed_tolerance for i in range(len(pred_bouts_vel[pid]))]) for pid in range(self.n_preds)]

        discarded_bouts_speed = [np.array(pred_bout_bounds_filtered[pid])[np.logical_not(pred_bout_speed_filters[pid])] for pid in range(self.n_preds)]
        pred_bout_bounds_filtered = [np.array(pred_bout_bounds_filtered[pid])[pred_bout_speed_filters[pid]] for pid in range(self.n_preds)]

        print(f"--> {np.sum([len(pred_bout_bounds_filtered[pid]) for pid in range(self.n_preds)])} remain")

        self.pred_bout_bounds = pred_bout_bounds_filtered
        pred_bout_bounds_filtered = list(itertools.chain.from_iterable(pred_bout_bounds_filtered))
        self.pred_datas_arr_bouts = [np.vstack([pred_data_arr[pred_bout_bound_filtered[0]:pred_bout_bound_filtered[1]] for pred_bout_bound_filtered in pred_bout_bounds_filtered]) for pred_data_arr in self.pred_datas_arr]
        self.agent_datas_arr_bouts = [np.vstack(
            [agent_data_arr[pred_bout_bound_filtered[0]:pred_bout_bound_filtered[1]] for pred_bout_bound_filtered in pred_bout_bounds_filtered]) for agent_data_arr in self.agent_datas_arr]
        self.timestamps_bouts = np.concatenate(
            [self.timestamps[pred_bout_bound_filtered[0]:pred_bout_bound_filtered[1]] for pred_bout_bound_filtered in pred_bout_bounds_filtered])
        self.agent_com_bouts = np.concatenate([self.agent_com[pred_bout_bound_filtered[0]:pred_bout_bound_filtered[1]] for pred_bout_bound_filtered in pred_bout_bounds_filtered])


        self.pred_bout_bounds_discarded = [discarded_bouts_length, discarded_bouts_speed]
        pred_bout_bounds_discarded = [list(itertools.chain.from_iterable(self.pred_bout_bounds_discarded[i])) for i in range(len(self.pred_bout_bounds_discarded))]

        self.pred_datas_arr_bouts_discarded = [[np.vstack(
            [pred_data_arr[pred_bout_bound_discarded[0]:pred_bout_bound_discarded[1]] for pred_bout_bound_discarded in
             pred_bout_bounds_discarded[i]]) for pred_data_arr in self.pred_datas_arr] for i in range(len(self.pred_bout_bounds_discarded))]
        self.agent_datas_arr_bouts_discarded = [[np.vstack(
            [agent_data_arr[pred_bout_bound_discarded[0]:pred_bout_bound_discarded[1]] for pred_bout_bound_discarded in
             pred_bout_bounds_discarded[i]]) for agent_data_arr in self.agent_datas_arr] for i in range(len(self.pred_bout_bounds_discarded))]
        self.timestamps_bouts_discarded = [np.concatenate(
            [self.timestamps[pred_bout_bound_discarded[0]:pred_bout_bound_discarded[1]] for pred_bout_bound_discarded in
             pred_bout_bounds_discarded[i]]) for i in range(len(self.pred_bout_bounds_discarded))]
        self.agent_com_bouts_discarded = [np.concatenate(
            [self.agent_com[pred_bout_bound_discarded[0]:pred_bout_bound_discarded[1]] for pred_bout_bound_discarded in
             pred_bout_bounds_discarded[i]]) for i in range(len(self.pred_bout_bounds_discarded))]


if __name__ == "__main__":
    #find CoBeHumanExperimentsData/ -name '*.zip' -exec sh -c 'unzip -d "${1%.*}" "$1"' _ {} \;

    import matplotlib.pyplot as plt
    round_id = "2837465091"
    round_types = ["P1", "P2", "Shared", "P1R1", "P1R2", "P1R3"]
    round_type_id = 3
    file_path = f"./CoBeHumanExperimentsData/{round_id}/{round_types[round_type_id-1] if round_type_id < 4 else round_types[round_type_id-1][:2].upper()}/{round_id}_{round_types[round_type_id-1][:2].upper() if round_type_id < 4 else round_types[round_type_id-1]}.csv"

    if not os.path.exists(file_path):
        file_path = f"./CoBeHumanExperimentsData/{round_id}/{round_types[round_type_id-1][:2].upper()}/{round_id}_{round_types[round_type_id-1][:2].upper()}.csv"
        round_types[round_type_id - 1] = round_types[round_type_id -1][:2].upper()

    exp = Round(file_path, n_preds=2 if round_type_id == 3 else 1,
                dist_tolerance=0.7, margin=10, min_bout_length=30,
                speed_tolerance=0.15, speed_threshold=5.)

    exp_plotter = RoundPlotter(exp, mac=True)
    #exp_plotter.plot_metrics(time_window_dur=2, smoothing_args={'kernel': gaussian, 'window_size': 100},
    #                         #save=True, # saving does not seem to work yet
    #                         out_file_path=f"./CoBeHumanExperimentsDataAnonymized/{round_id}/{round_types[round_type_id-1]}/{round_id}_{round_types[round_type_id-1][:2].upper()}.mp4",
    #                         com_only=True)
    #exp_plotter.plot_predator_acc_smoothings(time_window_dur=2, window_size=40, com_only=True)
    #exp_plotter.plot_bout_trajectories(discarded=False)
    #exp_plotter.plot_bout_trajectories(discarded=True, filter_number=0)  # minimum bout length filter
    #exp_plotter.plot_bout_trajectories(discarded=True, filter_number=1)  # speed filter
    exp_plotter.plot_bout_division(com_only=True, separate_predators=True)
