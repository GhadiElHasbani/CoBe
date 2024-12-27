import itertools
from tinyflux import TinyFlux
from helpers import *
from round_plotter import RoundPlotter
from typing import Dict
import os

class Round:

    def __init__(self,
                 file_path: str,
                 T_start: int = 2000,
                 T: int = 4000,
                 n_agents: int = 50,
                 n_preds: int = 1,
                 center: Tuple[float, float] = (0, 0),
                 radius: float = 20,
                 remove_dup_timevals: bool = False,
                 min_bout_length: int = 2,
                 infer_timestamps: bool = True,
                 write_timestamps_to_csv: bool = False,
                 tolerance: float = 0.9,
                 margin: int = 0):
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

        # retrieving timestamp data to determine time of each observation
        if infer_timestamps:
            self.timestamps = infer_timestamps_from_json_files(file_path=self.file_path)
        else:
            self.timestamps = self.db_get_field_values("t")

        if write_timestamps_to_csv:
            write_column_to_csv(file_path=self.file_path, column=self.timestamps, column_name='f_t')

        self.timestamps = [t - min(self.timestamps) for t in self.timestamps]
        print(max(self.timestamps) - min(self.timestamps))
        # retrieving data for a single agent
        self.agent_datas = []
        self.agent_datas_arr = []

        n_missing = 0
        for agent_id in range(self.n_agents):
            agent_x = self.db.get_field_values("x" + str(agent_id))
            agent_y = self.db.get_field_values("y" + str(agent_id))
            agent_data = [(agent_x[i], agent_y[i]) for i in range(len(timesteps))]
            # sorting agent data according to timesteps list
            agent_data = [agent_data[i] for i in sorted(range(len(timesteps)), key=lambda x: timesteps[x])]
            # slicing to length
            agent_data = agent_data[self.T_start:self.T]
            agent_data_arr = convert_list_of_tuples_to_array(agent_data)
            n_missing += np.sum(np.isnan(agent_data_arr).astype('int') + np.isinf(agent_data_arr).astype('int') > 0)
            self.agent_datas.append(agent_data)
            self.agent_datas_arr.append(agent_data_arr)
        print(f"---> {n_missing} missing values for agents")

        # retrieving data for predator agent(s)
        self.pred_datas = []
        self.pred_datas_arr = []

        n_missing = 0
        for pred_id in range(self.n_preds):

            pred_x = self.db.get_field_values("prx" + str(pred_id))
            pred_y = self.db.get_field_values("pry" + str(pred_id))
            pred_data = [(pred_x[i], pred_y[i]) for i in range(len(timesteps))]
            # sorting pred data according to timesteps list
            pred_data = [pred_data[i] for i in sorted(range(len(timesteps)), key=lambda x: timesteps[x])]
            # slicing to length
            pred_data = pred_data[self.T_start:self.T]
            pred_data_arr = convert_list_of_tuples_to_array(pred_data)
            n_missing += np.sum(np.isnan(pred_data_arr).astype('int') + np.isinf(pred_data_arr).astype('int') > 0)
            self.pred_datas.append(pred_data)
            self.pred_datas_arr.append(pred_data_arr)
        print(f"---> {n_missing} missing values for predator(s)")

        # sorting and slicing timesteps and timestamps
        self.timesteps = np.array(sorted(timesteps)[T_start:T])
        self.timestamps = np.array(sorted(self.timestamps)[T_start:T])
        print(max(self.timestamps) - min(self.timestamps))
        plt.hist(np.diff(self.timestamps))
        plt.show()
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
        self.min_bout_length = min_bout_length
        self.pred_datas_arr_bouts = None
        self.agent_datas_arr_bouts = None
        self.timestamps_bouts = None
        self.agent_com_bouts = None
        self.segment_into_bouts(tolerance=tolerance, margin=margin)

    def compute_speed(self, datas_arr: List[NDArray]) -> List[NDArray[float]]:
        return [
            np.insert(np.sqrt(np.sum(np.diff(data_arr, axis=0) ** 2, axis=1)) / (np.diff(self.timestamps)), 0, [0.,]) for
            data_arr in datas_arr]

    def compute_agent_speed(self) -> List[NDArray[float]]:
        agent_datas_vel = self.compute_speed(self.agent_datas_arr)
        n_missing = np.sum(
            [np.sum(np.isnan(agent_data_vel).astype('int') + np.isinf(agent_data_vel).astype('int') > 0) for agent_data_vel
             in agent_datas_vel])
        print(f"---> {n_missing} missing values for predator speed")
        return agent_datas_vel

    def compute_predator_speed(self) -> List[NDArray[float]]:
        pred_datas_vel = self.compute_speed(self.pred_datas_arr)
        n_missing = np.sum(
            [np.sum(np.isnan(pred_data_vel).astype('int') + np.isinf(pred_data_vel).astype('int') > 0) for pred_data_vel
             in pred_datas_vel])
        print(f"---> {n_missing} missing values for predator speed")
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
        n_missing = np.sum([np.sum(np.isnan(pred_data_acc).astype('int') + np.isinf(pred_data_acc).astype('int') > 0) for pred_data_acc in pred_datas_acc])
        print(f"---> {n_missing} missing values for predator acceleration")
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

    def segment_into_bouts(self, tolerance: float = 0.9, margin: int = 0):
        threshold = tolerance*self.width/2

        print(f"Segmenting bouts using threshold of {threshold} on predator distance from arena center {self.center}")
        preds_dist_to_center = self.compute_predator_distance_to_center()

        preds_points_during_bouts = [pred_dist_to_center < threshold for pred_dist_to_center in preds_dist_to_center]
        temp = [get_bounds(pred_points_during_bouts, margin=margin) for pred_points_during_bouts in preds_points_during_bouts]
        preds_bout_bounds = list(itertools.chain.from_iterable([temp_el[0] for temp_el in temp]))
        preds_bout_lengths = list(itertools.chain.from_iterable([temp_el[1] for temp_el in temp]))
        preds_bout_bounds_filtered = np.array(preds_bout_bounds)[np.array(preds_bout_lengths) >= self.min_bout_length].tolist()
        self.pred_datas_arr_bouts = [np.vstack([pred_data_arr[preds_bout_bound_filtered[0]:preds_bout_bound_filtered[1]] for preds_bout_bound_filtered in preds_bout_bounds_filtered]) for pred_data_arr in self.pred_datas_arr]
        self.agent_datas_arr_bouts = [np.vstack(
            [agent_data_arr[preds_bout_bound_filtered[0]:preds_bout_bound_filtered[1]] for preds_bout_bound_filtered in
             preds_bout_bounds_filtered]) for agent_data_arr in self.agent_datas_arr]
        self.timestamps_bouts = np.concatenate(
            [self.timestamps[preds_bout_bound_filtered[0]:preds_bout_bound_filtered[1]] for preds_bout_bound_filtered in
             preds_bout_bounds_filtered])
        self.agent_com_bouts = np.concatenate([self.agent_com[preds_bout_bound_filtered[0]:preds_bout_bound_filtered[1]] for preds_bout_bound_filtered in
             preds_bout_bounds_filtered])


if __name__ == "__main__":
    #find CoBeHumanExperimentsData/ -name '*.zip' -exec sh -c 'unzip -d "${1%.*}" "$1"' _ {} \;

    import matplotlib.pyplot as plt
    round_id = "2837465091"
    round_types = ["P1", "P2", "Shared", "P1R1", "P1R2", "P1R3"]
    round_type_id = 3
    file_path = f"./CoBeHumanExperimentsData/{round_id}/{round_types[round_type_id-1] if round_type_id < 4 else round_types[round_type_id-1][:2].upper()}/{round_id}_{round_types[round_type_id-1][:2].upper() if round_type_id < 4 else round_types[round_type_id-1]}.csv"
    print(file_path)
    if not os.path.exists(file_path):
        file_path = f"./CoBeHumanExperimentsData/{round_id}/{round_types[round_type_id-1][:2].upper()}/{round_id}_{round_types[round_type_id-1][:2].upper()}.csv"
        round_types[round_type_id - 1] = round_types[round_type_id -1][:2].upper()

    exp = Round(file_path, n_preds=2 if round_type_id == 3 else 1, tolerance=0.7, margin=10)
    #import matplotlib.pyplot as plt
    #plt.hist(np.diff(exp.timesteps), bins=50)
    #plt.show()
    #plt.hist(np.diff(exp.timestamps), bins=50)
    #plt.show()
    print(np.mean(np.diff(exp.timestamps)))
    exp_plotter = RoundPlotter(exp, mac=True)
    exp_plotter.plot_metrics(time_window_dur=0.07, smoothing_args={'kernel': gaussian, 'window_size': 50},
                             #save=True, # saving does not seem to work yet
                             out_file_path=f"./CoBeHumanExperimentsDataAnonymized/{round_id}/{round_types[round_type_id-1]}/{round_id}_{round_types[round_type_id-1][:2].upper()}.mp4",
                             com_only=True,
                             max_abs_speed=0.01)
    exp_plotter.plot_predator_acc_smoothings(time_window_dur=0.07, window_size=40, com_only=True)
    exp_plotter.plot_bouts()