from round import *
import os
from typing import Dict, Any


class RoundsManager:

    def __init__(self,
                 input_path: str,
                 output_path: str):
        self.input_path = input_path
        self.output_path = output_path
        self.round_types = ["P1", "P2", "Shared", "P1R1", "P1R2", "P1R3"]

        self.experiment_ids = {}

    def write_bout_info(self,
                        exp_args: Dict[str, Any],
                        exp_plotter_args: Dict[str, Any],
                        mark_speed_spike: bool = False,
                        speed_spike_threshold: float = 10.):
        experiment_ids = next(os.walk(self.input_path))[1]
        for experiment_id in experiment_ids:
            exp_output_path = f"{self.output_path}/{experiment_id}"
            if not os.path.exists(exp_output_path):
                os.makedirs(exp_output_path)

            rounds = next(os.walk(f"{self.input_path}/{experiment_id}"))[1]

            if len(rounds) == 1:
                round_type_ids = [3, 4, 5]
            else:
                round_type_ids = [0, 1, 2]

            self.experiment_ids[experiment_id] = round_type_ids

            for round_type_id in round_type_ids:
                round_output_path = f"{exp_output_path}/{self.round_types[round_type_id]}"
                if not os.path.exists(round_output_path):
                    os.makedirs(round_output_path)

                file_path = f"{self.input_path}/{experiment_id}/{self.round_types[round_type_id] if round_type_id < 4 else self.round_types[round_type_id][:2].upper()}/{experiment_id}_{self.round_types[round_type_id][:2].upper() if round_type_id < 4 else self.round_types[round_type_id]}.csv"

                if not os.path.exists(file_path):
                    file_path = f"{self.input_path}/{experiment_id}/{self.round_types[round_type_id][:2].upper()}/{experiment_id}_{self.round_types[round_type_id][:2].upper()}.csv"

                if not os.path.exists(file_path):
                    file_path = f"{self.input_path}/{experiment_id}/{self.round_types[round_type_id][:2].upper()}/{experiment_id}_{self.round_types[round_type_id][:2]}_{self.round_types[round_type_id][2:]}.csv"

                if not os.path.exists(file_path):
                    file_path = f"{self.input_path}/{experiment_id}/{self.round_types[round_type_id][:2].upper()}/{experiment_id}_{self.round_types[round_type_id]}.csv"

                if round_type_id == 5 and not os.path.exists(file_path):
                    pass
                else:
                    exp = Round(file_path, n_preds=2 if round_type_id == 2 else 1, **exp_args)

                    exp.write_bout_info(output_path=round_output_path, exp_plotter_args=exp_plotter_args,
                                        mark_speed_spike=mark_speed_spike, speed_spike_threshold=speed_spike_threshold)



if __name__ == "__main__":

    rm = RoundsManager(input_path="./CoBeHumanExperimentsData", output_path="./bout_info")

    exp_args = {'center': (0, 0),
                'dist_tolerance': 0.7, 'margin': 10, 'min_bout_length': 30,
                'speed_tolerance': 0.15, 'speed_threshold': 0.475, 'absolute_speed_threshold': 6.65,
                'bout_ids_to_remove': None, 'evasion_pred_dist_to_com_limit': None, 'evasion_vel_angle_change_limit': None}

    exp_plotter_args = {'mac': True}

    rm.write_bout_info(exp_args=exp_args, exp_plotter_args=exp_plotter_args)
