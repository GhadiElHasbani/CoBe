from typing import Dict, Union
import matplotlib.pyplot as plt
import matplotlib.gridspec as GridSpec
from matplotlib import animation
import matplotlib.patches as patches
import matplotlib as mpl
import mpl_toolkits.mplot3d.art3d as art3d

from helpers import *


class RoundPlotter:

    def __init__(self, round, mac: bool = False):
        self.round = round
        self.colors = ['Reds', 'Blues', 'Yellows']

        if mac:
            mpl.use('macosx')

    def update_trajectories_ax_specs(self, ax: plt.Axes, t: int) -> None:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("Time [s]")
        ax.set_title(f"Trajectory of {self.round.n_agents} P-Points and {self.round.n_preds} Predator(s)")
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)

        ax.set_zlim(self.round.timestamps[0], self.round.timestamps[t])

    def update_arena_borders(self, ax: plt.Axes, z: float) -> None:
        for i in ["z"]:
            circle = plt.Circle(self.round.center, self.round.radius, fill=False, linewidth=1, alpha=1, linestyle="--")
            rect = patches.Rectangle((-self.round.width / 2, -self.round.width / 2), self.round.width, self.round.width,
                                     linewidth=1, fill=False, linestyle=":")
            ax.add_patch(rect)
            ax.add_artist(circle)
            art3d.pathpatch_2d_to_3d(rect, z=z, zdir=i)
            art3d.pathpatch_2d_to_3d(circle, z=z, zdir=i)


    def plot_agent_trajectories(self, agent_data: Union[List[Tuple[float, float]], NDArray[float]], ax: plt.Axes, t: int, cmap: str = 'viridis'):
        ax.scatter([x[0] for x in agent_data][:t], [x[1] for x in agent_data][:t], self.round.timestamps[:t],
                   c=range(t),
                   cmap=cmap, s=0.1)

    def update_agent_trajectories(self, agent_datas: List[List[Tuple[float, float]]], ax: plt.Axes, t: int, z: float,
                                  com_only: bool = False, agent_com: NDArray[float] = None, max_n_agents: int = None) -> None:
        def plot_agent_markers(agent_data: Union[List[Tuple[float, float]], NDArray[float]], ax: plt.Axes, t: int, z: float, marker: str='o', s:int=25,
                               c: str = None):
            x = agent_data[t][0]
            y = agent_data[t][1]

            ax.scatter(x, y, z, s=s, marker=marker, c=c)

        if com_only:
            plot_agent_markers(agent_com, ax=ax, t=t, z=z, marker='x', s=100, c='black')
            self.plot_agent_trajectories(agent_com, ax=ax, t=t, cmap='Greys')

        for aid in range(min([max_n_agents if max_n_agents is not None else self.round.n_agents, self.round.n_agents])):
            agent_data = agent_datas[aid]
            plot_agent_markers(agent_data, ax=ax, t=t, z=z)

            if not com_only:
                self.plot_agent_trajectories(agent_data, ax=ax, t=t)

    def update_predator_trajectories(self, pred_datas: List[List[Tuple[float, float]]], ax: plt.Axes, t: int, z: float, labels: List[str] = None) -> None:
        for pid in range(self.round.n_preds):
            pred_data = pred_datas[pid]

            # showing predator trajectory with some colormap
            ax.scatter([x[0] for x in pred_data][:t], [x[1] for x in pred_data][:t], self.round.timestamps[:t],
                       c=range(t),
                       cmap=self.colors[pid],
                       s=0.5)

            x = pred_data[t][0]
            y = pred_data[t][1]

            ax.scatter(x, y, z, s=100, marker='x', color=self.colors[pid][:-1], label=labels[pid] if labels is not None else None)

        if labels is not None:
            ax.legend()

    def update_trajectories(self, agent_datas: List[List[Tuple[float, float]]], pred_datas: List[List[Tuple[float, float]]],
                            ax: plt.Axes, t: int, z: float, pred_labels: List[str] = None,
                            show_arena_borders: bool = True, com_only: bool = False,
                            agent_com=NDArray[float]) -> None:
        self.update_trajectories_ax_specs(ax, t=t)

        if show_arena_borders:
            # Arena borders
            self.update_arena_borders(ax=ax, z=z)

        # agent data
        self.update_agent_trajectories(agent_datas=agent_datas, ax=ax, t=t, z=z, com_only=com_only, agent_com=agent_com)

        # predator data
        self.update_predator_trajectories(pred_datas=pred_datas, ax=ax, t=t, z=z, labels=pred_labels)

    def update_metric_ax_specs(self, data_list: List[NDArray[float]], ax: plt.Axes,
                               t: int, t_start: int, time_window_dur: float,
                               metric: str, max_abs_val: float = None):
        if max_abs_val is None:
            max_abs_val = np.nanmax([np.nanmax(abs(data)) for data in data_list])
        ax.set_ylim(np.nanmax([np.nanmin([np.nanmin(data[t_start:t + 1]) for data in data_list]), -max_abs_val]),
                    np.nanmin([np.nanmax([np.nanmax(data[t_start:t + 1]) for data in data_list]), max_abs_val]))
        ax.set_xlim(self.round.timestamps[t] - 2 * time_window_dur / 3,
                    self.round.timestamps[t] + time_window_dur / 3)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(metric.split(' ')[0])
        ax.set_title(f"Predator {metric} over time")

    def update_predator_lines(self, data_list: List[NDArray[float]], ax: plt.Axes, lines: List[plt.Line2D],
                              t: int, t_start: int, linestyle: str = '-', alpha: float = 1., colors: List[str] = None) -> None:
        if colors is None:
            colors = self.colors

        for pid in range(len(data_list)):
            data = data_list[pid]
            lines[pid], = ax.plot([], [], color=colors[pid][:-1], linestyle=linestyle, alpha=alpha)
            lines[pid].set_data(self.round.timestamps[t_start:t + 1], data[t_start:t + 1])


    def plot_metrics(self, time_window_dur: float = 2000., smoothing_args: Dict = None,
                     max_abs_speed: float = None, max_abs_acceleration: float = None,
                     com_only: bool = False,
                     save: bool = False, out_file_path: str = "metrics.mp4") -> None:
        def update(t: int, args_dict: Dict) -> Dict:
            t_start = max([0, find_nearest(self.round.timestamps, self.round.timestamps[t] - 2*time_window_dur / 3)])
            z = self.round.timestamps[t]

            args_dict['ax'].cla()
            args_dict['ax_vel'].cla()
            args_dict['ax_acc'].cla()
            args_dict['ax_com'].cla()
            args_dict['ax_bor'].cla()
            args_dict['ax_att'].cla()

            self.update_trajectories(agent_datas=self.round.agent_datas, pred_datas=self.round.pred_datas,
                                     ax=args_dict['ax'], t=t, z=z, com_only=com_only, agent_com=self.round.agent_com)

            # Speed axes
            self.update_predator_lines(pred_datas_vel, ax=args_dict['ax_vel'], lines=[args_dict[f'vel_{pid}'] for pid in range(self.round.n_preds)],
                                       t=t, t_start=t_start)
            self.update_metric_ax_specs(pred_datas_vel, ax=args_dict['ax_vel'],
                                        t=t, t_start=t_start, time_window_dur=time_window_dur,
                                        metric="speed", max_abs_val=max_abs_speed)
            # Acceleration axes
            self.update_predator_lines(pred_datas_acc_smoothed, ax=args_dict['ax_acc'],
                                       lines=[args_dict[f'acc_smoothed_{pid}'] for pid in range(self.round.n_preds)],
                                       t=t, t_start=t_start)
            self.update_predator_lines(pred_datas_acc, ax=args_dict['ax_acc'],
                                       lines=[args_dict[f'acc_{pid}'] for pid in range(self.round.n_preds)],
                                       t=t, t_start=t_start, linestyle='--', alpha=0.6)
            self.update_metric_ax_specs(vcombine_lists_of_arrays([pred_datas_acc, pred_datas_acc_smoothed]), ax=args_dict['ax_acc'],
                                        t=t, t_start=t_start, time_window_dur=time_window_dur,
                                        metric="acceleration", max_abs_val=max_abs_acceleration)
            # Predator distance to agents center of mass axes
            self.update_predator_lines(pred_datas_com, ax=args_dict['ax_com'],
                                       lines=[args_dict[f'com_{pid}'] for pid in range(self.round.n_preds)],
                                       t=t, t_start=t_start)
            self.update_metric_ax_specs(pred_datas_com, ax=args_dict['ax_com'],
                                        t=t, t_start=t_start, time_window_dur=time_window_dur,
                                        metric="distance from agents center of mass")
            # Predator distance to border axes
            self.update_predator_lines(pred_datas_bor, ax=args_dict['ax_bor'],
                                       lines=[args_dict[f'bor_{pid}'] for pid in range(self.round.n_preds)],
                                       t=t, t_start=t_start)
            self.update_metric_ax_specs(pred_datas_bor, ax=args_dict['ax_bor'],
                                        t=t, t_start=t_start, time_window_dur=time_window_dur,
                                        metric="distance from closest border")
            # Predator distance to border axes
            self.update_predator_lines(pred_datas_att, ax=args_dict['ax_att'],
                                       lines=[args_dict[f'att_{pid}'] for pid in range(self.round.n_preds)],
                                       t=t, t_start=t_start)
            self.update_metric_ax_specs(pred_datas_att, ax=args_dict['ax_att'],
                                        t=t, t_start=t_start, time_window_dur=time_window_dur,
                                        metric="attack angle")

            return args_dict

        fig = plt.figure(dpi=100)
        gs = GridSpec.GridSpec(5, 2)

        ## Trajectory figure
        ax = fig.add_subplot(gs[:, 0], projection='3d')

        ## Metric figures
        if smoothing_args is None:
            smoothing_args = {'smoothing_method': 'window',
                              'kernel': lambda x, i, b: 1,
                              'window_size': 40}
        # Compute metrics
        pred_datas_vel = self.round.compute_predator_speed()
        pred_datas_acc_smoothed = self.round.compute_predator_acceleration(smoothing_args=smoothing_args)
        pred_datas_acc = self.round.compute_predator_acceleration(smooth=False)
        pred_datas_com = self.round.compute_predator_distance_to_agent_com()
        pred_datas_bor = self.round.compute_predator_distance_to_border()
        pred_datas_att = self.round.compute_predator_attack_angle()

        # speed
        ax_vel = fig.add_subplot(gs[0, 1])
        # acceleration
        ax_acc = fig.add_subplot(gs[1, 1])
        # predator distance to agents center of mass
        ax_com = fig.add_subplot(gs[2, 1])
        # predator distance from closest border
        ax_bor = fig.add_subplot(gs[3, 1])
        # predator distance from closest border
        ax_att = fig.add_subplot(gs[4, 1])

        args_dict = {'ax': ax,
                     'ax_vel': ax_vel,
                     'ax_acc': ax_acc,
                     'ax_com': ax_com,
                     'ax_bor': ax_bor,
                     'ax_att': ax_att}

        for pid in range(self.round.n_preds):
            args_dict[f'vel_{pid}'], = ax_vel.plot([], [], color=self.colors[pid][:-1])
            args_dict[f'acc_{pid}'], = ax_acc.plot([], [], color=self.colors[pid][:-1])
            args_dict[f'acc_smoothed_{pid}'], = ax_acc.plot([], [], color=self.colors[pid][:-1])
            args_dict[f'com_{pid}'], = ax_com.plot([], [], color=self.colors[pid][:-1])
            args_dict[f'bor_{pid}'], = ax_bor.plot([], [], color=self.colors[pid][:-1])
            args_dict[f'att_{pid}'], = ax_att.plot([], [], color=self.colors[pid][:-1])

        ani = animation.FuncAnimation(fig=fig, func=update, frames=len(self.round.timestamps), interval=1, fargs=(args_dict,))

        plt.tight_layout()
        plt.show()

        if save:
            # saving to m4 using ffmpeg writer
            ani.save(out_file_path, writer='ffmpeg')
            plt.close()

    def plot_predator_acc_smoothings(self, time_window_dur: float = 2000., window_size: int = 40,
                                     com_only: bool = True,
                                     save: bool = False, out_file_path: str = "acc_smoothings.mp4") -> None:
        def update(t: int, args_dict: Dict) -> Dict:
            t_start = max([0, find_nearest(self.round.timestamps, self.round.timestamps[t] - 2 * time_window_dur / 3)])
            z = self.round.timestamps[t]

            args_dict['ax'].cla()
            args_dict['ax_acc'].cla()
            args_dict['ax_acc_w'].cla()
            args_dict['ax_acc_w_gaus'].cla()
            args_dict['ax_acc_bw'].cla()
            args_dict['ax_dts'].cla()

            self.update_trajectories(agent_datas=self.round.agent_datas, pred_datas=self.round.pred_datas,
                                     ax=args_dict['ax'], t=t, z=z, com_only=com_only, agent_com=self.round.agent_com)

            # Raw predator acceleration
            self.update_predator_lines(pred_datas_acc, ax=args_dict['ax_acc'],
                                       lines=[args_dict[f'acc_{pid}'] for pid in range(self.round.n_preds)],
                                       t=t, t_start=t_start)
            self.update_metric_ax_specs(pred_datas_acc, ax=args_dict['ax_acc'],
                                        t=t, t_start=t_start, time_window_dur=time_window_dur,
                                        metric=f"acceleration (raw)")
            # Predator acceleration with average smoothing kernel
            self.update_predator_lines(pred_datas_acc_w, ax=args_dict['ax_acc_w'],
                                           lines=[args_dict[f'acc_w_{pid}'] for pid in range(self.round.n_preds)],
                                           t=t, t_start=t_start)
            self.update_predator_lines(pred_datas_acc, ax=args_dict['ax_acc_w'],
                                       lines=[args_dict[f'acc_w_raw_{pid}'] for pid in range(self.round.n_preds)],
                                       t=t, t_start=t_start, linestyle='--', alpha=0.6)
            self.update_metric_ax_specs(vcombine_lists_of_arrays([pred_datas_acc, pred_datas_acc_w]), ax=args_dict['ax_acc_w'],
                                        t=t, t_start=t_start, time_window_dur=time_window_dur,
                                        metric=f"acceleration (average)")
            # Predator acceleration with gaussian smoothing kernel
            self.update_predator_lines(pred_datas_acc_w_gaus, ax=args_dict['ax_acc_w_gaus'],
                                           lines=[args_dict[f'acc_w_gaus_{pid}'] for pid in range(self.round.n_preds)],
                                           t=t, t_start=t_start)
            self.update_predator_lines(pred_datas_acc, ax=args_dict['ax_acc_w_gaus'],
                                       lines=[args_dict[f'acc_w_gaus_raw_{pid}'] for pid in range(self.round.n_preds)],
                                       t=t, t_start=t_start, linestyle='--', alpha=0.6)
            self.update_metric_ax_specs(vcombine_lists_of_arrays([pred_datas_acc, pred_datas_acc_w_gaus]), ax=args_dict['ax_acc_w_gaus'],
                                        t=t, t_start=t_start, time_window_dur=time_window_dur,
                                        metric=f"acceleration (gaussian)")
            # Predator acceleration with butterworth filter
            self.update_predator_lines(pred_datas_acc_bw, ax=args_dict['ax_acc_bw'],
                                           lines=[args_dict[f'acc_bw_{pid}'] for pid in range(self.round.n_preds)],
                                           t=t, t_start=t_start)
            self.update_predator_lines(pred_datas_acc, ax=args_dict['ax_acc_bw'],
                                       lines=[args_dict[f'acc_bw_raw_{pid}'] for pid in range(self.round.n_preds)],
                                       t=t, t_start=t_start, linestyle='--', alpha=0.6)
            self.update_metric_ax_specs(vcombine_lists_of_arrays([pred_datas_acc, pred_datas_acc_bw]), ax=args_dict['ax_acc_bw'],
                                        t=t, t_start=t_start, time_window_dur=time_window_dur,
                                        metric=f"acceleration \n (butterworth)")
            # dts
            self.update_predator_lines([np.concatenate([np.array([0]), dts])], ax=args_dict['ax_dts'],
                                      lines=[args_dict['dts']],
                                       t=t, t_start=t_start, colors=['Blacks'])
            self.update_metric_ax_specs([np.concatenate([np.array([0]), dts])], ax=args_dict['ax_dts'],
                                        t=t, t_start=t_start, time_window_dur=time_window_dur,
                                        metric="dt [s]")

            return args_dict

        fig = plt.figure(dpi=100)
        gs = GridSpec.GridSpec(5, 2)

        dts = np.diff(self.round.timestamps)

        ## Trajectory figure
        ax = fig.add_subplot(gs[:, 0], projection='3d')

        ## Metric figures
        # Compute metrics
        pred_datas_acc = self.round.compute_predator_acceleration(smooth=False)
        pred_datas_acc_w = self.round.compute_predator_acceleration(smooth=True,
                                                                    smoothing_args={'smoothing_method': 'window',
                                                                                        'kernel': lambda x, i, b: 1,
                                                                                        'window_size': window_size})
        pred_datas_acc_w_gaus = self.round.compute_predator_acceleration(smooth=True, smoothing_args={
                'smoothing_method': 'window',
                'kernel': gaussian, 'window_size': window_size})
        pred_datas_acc_bw = self.round.compute_predator_acceleration(smooth=True, smoothing_args={
                'smoothing_method': 'butterworth', 'fs': self.round.avg_fs/2, 'bw_fstart': 5, 'bw_fstop': 10, 'bw_order': 3})

        # speed
        ax_acc = fig.add_subplot(gs[0, 1])
        # acceleration
        ax_acc_w = fig.add_subplot(gs[1, 1])
        # predator distance to agents center of mass
        ax_acc_w_gaus = fig.add_subplot(gs[2, 1])
        # predator distance from closest border
        ax_acc_bw = fig.add_subplot(gs[3, 1])
        # time diffs
        ax_dts = fig.add_subplot(gs[4, 1])

        args_dict = {'ax': ax,
                         'ax_acc': ax_acc,
                         'ax_acc_w': ax_acc_w,
                         'ax_acc_w_gaus': ax_acc_w_gaus,
                         'ax_acc_bw': ax_acc_bw,
                         'ax_dts': ax_dts}

        for pid in range(self.round.n_preds):
            args_dict[f'acc_{pid}'], = ax_acc.plot([], [], color=self.colors[pid][:-1])
            args_dict[f'acc_w_{pid}'], = ax_acc_w.plot([], [], color=self.colors[pid][:-1])
            args_dict[f'acc_w_gaus_{pid}'], = ax_acc_w_gaus.plot([], [], color=self.colors[pid][:-1])
            args_dict[f'acc_bw_{pid}'], = ax_acc_bw.plot([], [], color=self.colors[pid][:-1])

            args_dict[f'acc_w_raw_{pid}'], = ax_acc_w.plot([], [], color=self.colors[pid][:-1])
            args_dict[f'acc_w_gaus_raw_{pid}'], = ax_acc_w_gaus.plot([], [], color=self.colors[pid][:-1])
            args_dict[f'acc_bw_raw_{pid}'], = ax_acc_bw.plot([], [], color=self.colors[pid][:-1])

        args_dict[f'dts'], = ax_dts.plot([], [])

        ani = animation.FuncAnimation(fig=fig, func=update, frames=len(self.round.timestamps), interval=1, fargs=(args_dict,))

        plt.tight_layout()
        plt.show()

        if save:
            # saving to m4 using ffmpeg writer
            ani.save(out_file_path, writer='ffmpeg')
            plt.close()

    def plot_bout_trajectories(self, com_only: bool = True, discarded: bool = False, filter_number: int = 0):
        def update(t: int, args_dict: Dict) -> Dict:
            z = self.round.timestamps[t]

            args_dict['ax'].cla()

            self.update_trajectories(agent_datas=agent_datas,
                                     pred_datas=pred_datas,
                                     ax=args_dict['ax'], t=t, z=z,
                                     com_only=com_only, agent_com=agent_com)

            return args_dict

        fig = plt.figure(dpi=100)
        gs = GridSpec.GridSpec(1, 1)

        ## Trajectory figure
        ax = fig.add_subplot(gs[0, 0], projection='3d')

        args_dict = {'ax': ax}
        agent_datas = self.round.agent_datas_arr_bouts_discarded[filter_number] if discarded else self.round.agent_datas_arr_bouts
        agent_com = self.round.agent_com_bouts_discarded[filter_number] if discarded else self.round.agent_com_bouts
        pred_datas = self.round.pred_datas_arr_bouts_discarded[filter_number] if discarded else self.round.pred_datas_arr_bouts
        timestamps = self.round.timestamps_bouts_discarded[filter_number] if discarded else self.round.timestamps_bouts

        ani = animation.FuncAnimation(fig=fig, func=update, frames=len(timestamps), interval=1, fargs=(args_dict,))

        plt.tight_layout()
        plt.show()

    def plot_bout_division_hbars(self, ax: plt.Axes = None, set_y_labels: bool = True, separate_predators: bool = True):
        y_pos = ['Predator 1', 'Predator 2', 'Predator 3'] if set_y_labels else np.arange(3) + 1 if separate_predators else np.full(3, 1)

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        min_bout_start = self.round.timestamps[-1]
        max_bout_end = self.round.timestamps[0]
        for pid in range(self.round.n_preds):
            left = self.round.timestamps[0]
            for i, (bout_start, bout_end) in enumerate(self.round.pred_bout_bounds[pid]):
                if separate_predators:
                    ax.barh(y_pos[pid], width=self.round.timestamps[bout_start] - left, left=left, color='grey')
                else:
                    min_bout_start = np.min([min_bout_start, self.round.timestamps[bout_start]])
                    max_bout_end = np.max([max_bout_end, self.round.timestamps[bout_end]])
                left = self.round.timestamps[bout_end]
                ax.barh(y_pos[pid], width=self.round.timestamps[bout_end] - self.round.timestamps[bout_start], left=self.round.timestamps[bout_start], color=self.colors[pid][:-1], alpha=0.6 if not separate_predators else 1)
            if separate_predators:
                ax.barh(y_pos[pid], width=self.round.timestamps[-1] - left, left=left, color='grey')
        if not separate_predators:
            ax.barh(y_pos[0], width=self.round.timestamps[-1] - max_bout_end, left=max_bout_end, color='grey')
            ax.barh(y_pos[0], width=min_bout_start - self.round.timestamps[0], left=self.round.timestamps[0], color='grey')
        ax.set_xlabel("Time [s]")
        ax.set_yticks(y_pos[:self.round.n_preds] if set_y_labels else [])
        ax.set_title("Predator bouts (colored) timeline")
        if ax is None:
            plt.show()

    def plot_bout_division(self, com_only: bool = False, separate_predators: bool = False,
                           save: bool = False, out_file_path: str = "bout_divisions.mp4") -> None:
        def update(t: int, args_dict: Dict) -> Dict:

            args_dict['ax'].cla()
            args_dict['ax_bout_div'].cla()

            if t == 0:
                args_dict['in_bout'] = [t in pred_bout_starts[pid] for pid in range(self.round.n_preds)]

            for pid in range(self.round.n_preds):
                if args_dict['in_bout'][pid]:
                    args_dict['in_bout'][pid] = args_dict['in_bout'][pid] and (t not in pred_bout_ends[pid])
                else:
                    args_dict['in_bout'][pid] = t in pred_bout_starts[pid]

            self.update_trajectories(agent_datas=self.round.agent_datas, pred_datas=self.round.pred_datas, pred_labels=[".", "."],
                                     ax=args_dict['ax'], t=t, z=self.round.timestamps[t], com_only=com_only, agent_com=self.round.agent_com)

            pred_labels = []
            for pid in range(self.round.n_preds):
                if args_dict['in_bout'][pid]:
                    pred_labels.append("in bout")
                else:
                    pred_labels.append("")
                args_dict['bout_indicator'][pid] = mpl.patches.Ellipse((50 + pid*50, self.round.n_preds + 0.75 if separate_predators else 1.5),
                                                                        30, height=0.05, color=self.colors[pid][:-1],
                                                                       clip_on=False, alpha=1. if args_dict['in_bout'][pid] else 0.4)
                args_dict['ax_bout_div'].add_patch(args_dict['bout_indicator'][pid])

            args_dict['legend'] = args_dict['ax'].legend(loc='upper right')
            [args_dict['legend'].get_texts()[pid].set_text(pred_labels[pid]) for pid in range(self.round.n_preds)]

            # Bout divison axes
            self.plot_bout_division_hbars(ax=args_dict['ax_bout_div'], set_y_labels=False, separate_predators=separate_predators)
            # add time bar
            args_dict['ax_bout_div'].axvline(x=self.round.timestamps[t], ymin=-0.1, ymax=(self.round.n_preds if separate_predators else 1)*0.88)
            args_dict['ax_bout_div'].set_xlim([0, self.round.timestamps[-1]])

            return args_dict

        fig = plt.figure(dpi=100)
        gs = GridSpec.GridSpec(1, 2)

        ## Trajectory figure
        ax = fig.add_subplot(gs[:, 0], projection='3d')

        ax_bout_div = fig.add_subplot(gs[0, 1])

        args_dict = {'ax': ax, 'ax_bout_div': ax_bout_div,
                     'in_bout': [False for _ in range(self.round.n_preds)], 'bout_id': -1,
                     'legend': ax.legend(loc='upper right'), 'bout_indicator': [mpl.patches.Ellipse((50 + pid*50, 1.5), 30, height=0.05, color=self.colors[pid][:-1],
                                            clip_on=False, alpha=0.4) for pid in range(self.round.n_preds)]}


        args_dict['time_bar'] = args_dict['ax_bout_div'].axvline(self.round.timestamps[0])
        pred_bout_starts = [[self.round.pred_bout_bounds[pid][i][0] for i in range(len(self.round.pred_bout_bounds[pid]))] for pid in range(self.round.n_preds)]
        pred_bout_ends = [[self.round.pred_bout_bounds[pid][i][1] for i in range(len(self.round.pred_bout_bounds[pid]))] for pid in range(self.round.n_preds)]

        ani = animation.FuncAnimation(fig=fig, func=update, frames=len(self.round.timestamps), interval=1, fargs=(args_dict,))

        plt.tight_layout()
        plt.show()

        if save:
            # saving to m4 using ffmpeg writer
            ani.save(out_file_path, writer='ffmpeg')
            plt.close()
