from rounds_manager import RoundsManager

rm = RoundsManager(input_path="./CoBeHumanExperimentsData", output_path="./round_info")

exp_args = {'center': (0, 0),
            'dist_tolerance': 0.7, 'margin': 10, 'min_bout_length': 30, 'detect_isolated_individuals': False,
            'speed_tolerance': 0.15, 'speed_threshold': 0.475, 'absolute_speed_threshold': 6.65,
            'bout_ids_to_remove': None, 'evasion_pred_dist_to_com_limit': None, 'evasion_vel_angle_change_limit': None}

exp_plotter_args = {'mac': True, 'fps': 15}

rm.write_round_info(exp_args=exp_args, exp_plotter_args=exp_plotter_args)