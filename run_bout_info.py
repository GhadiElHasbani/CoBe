from rounds_manager import RoundsManager

rm = RoundsManager(input_path="./CoBeHumanExperimentsData", output_path="./bout_info")

exp_args = {'center': (0, 0),
            'dist_tolerance': 0.7, 'margin': 80, 'min_bout_length': 170,
            'speed_tolerance': 0.15, 'speed_threshold': 0.475, 'absolute_speed_threshold': 6.65,
            'bout_ids_to_remove': None, 'evasion_pred_dist_to_com_limit': None, 'evasion_vel_angle_change_limit': None}

exp_plotter_args = {'mac': True}

rm.write_bout_info(exp_args=exp_args, exp_plotter_args=exp_plotter_args)
