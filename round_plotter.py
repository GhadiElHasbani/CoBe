from typing import Union
import matplotlib.pyplot as plt
import matplotlib.gridspec as GridSpec
from matplotlib import animation
import matplotlib.patches as patches
import matplotlib as mpl
import mpl_toolkits.mplot3d.art3d as art3d

from helpers import *


class RoundPlotter:

    def __init__(self, round, mac: bool = False, fps: int = 8):
        self.round = round
        self.fps = fps
        self.colors = ['Reds', 'Blues', 'Yellows']

        if mac:
            mpl.use('macosx')

    def update_trajectories_ax_specs(self, ax: plt.Axes, t: int, force_2d: bool = False) -> None:
        if force_2d:
            ax.axis('equal')
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"Trajectory of {self.round.n_agents} P-Points and {self.round.n_preds} Predator(s)")
        ax.set_xlim(-self.round.width*2/3 + self.round.center[0], self.round.width*2/3 + self.round.center[0])
        ax.set_ylim(-self.round.width*2/3 + self.round.center[1], self.round.width*2/3 + self.round.center[1])

        if not force_2d:
            ax.set_zlabel("Time [s]")
            ax.set_zlim(self.round.timestamps[0], self.round.timestamps[t])

    def update_arena_borders(self, ax: plt.Axes, z: float, force_2d: bool = False) -> None:
        for i in ["z"]:
            circle = plt.Circle(self.round.center, self.round.radius, fill=False, linewidth=1, alpha=1, linestyle="--")
            rect = patches.Rectangle((-self.round.width / 2 + self.round.center[0], -self.round.width / 2 + self.round.center[1]), self.round.width, self.round.width,
                                     linewidth=1, fill=False, linestyle=":")
            ax.add_patch(rect)
            ax.add_artist(circle)
            if not force_2d:
                art3d.pathpatch_2d_to_3d(rect, z=z, zdir=i)
                art3d.pathpatch_2d_to_3d(circle, z=z, zdir=i)

    def plot_agent_trajectories(self, agent_data: Union[List[Tuple[float, float]], NDArray[float]], ax: plt.Axes, t: int,
                                cmap: str = 'viridis', force_2d: bool = False) -> None:
        if force_2d:
            ax.scatter([x[0] for x in agent_data][:t], [x[1] for x in agent_data][:t],
                       c=range(t), cmap=cmap, s=0.1)
        else:
            ax.scatter([x[0] for x in agent_data][:t], [x[1] for x in agent_data][:t], self.round.timestamps[:t],
                       c=range(t), cmap=cmap, s=0.1)

    def update_agent_trajectories(self, agents_data: List[List[Tuple[float, float]]], ax: plt.Axes, t: int, z: float,
                                  force_2d:bool=False, keep_agent_history: bool = False, keep_com_history: bool = True,
                                  show_com: bool = False, agent_com: NDArray[float] = None, max_n_agents: int = None) -> None:
        def plot_agent_markers(agent_data: Union[List[Tuple[float, float]], NDArray[float]], ax: plt.Axes, t: int, z: float, marker: str='o', s:int=25,
                               c: str = None, force_2d:bool=False):
            x = agent_data[t][0]
            y = agent_data[t][1]

            if force_2d:
                ax.scatter(x, y, s=s, marker=marker, c=c)
            else:
                ax.scatter(x, y, z, s=s, marker=marker, c=c)

        for aid in range(min([max_n_agents if max_n_agents is not None else self.round.n_agents, self.round.n_agents])):
            agent_data = agents_data[aid]
            plot_agent_markers(agent_data, ax=ax, t=t, z=z, force_2d=force_2d)

            if keep_agent_history:
                self.plot_agent_trajectories(agent_data, ax=ax, t=t, force_2d=force_2d)

        if show_com:
            plot_agent_markers(agent_com, ax=ax, t=t, z=z, marker='x', s=100, c='black', force_2d=force_2d)
            if keep_com_history:
                self.plot_agent_trajectories(agent_com, ax=ax, t=t, cmap='Greys', force_2d=force_2d)


    def update_predator_trajectories(self, preds_data: List[List[Tuple[float, float]]],
                                     ax: plt.Axes, t: int, z: float, labels: List[str] = None,
                                     show_pred_vel_vector: bool = False, preds_data_vel: List[NDArray[float]] = None,
                                     keep_history: bool = True, force_2d: bool = False) -> None:
        for pid in range(self.round.n_preds):
            pred_data = preds_data[pid]

            # showing predator trajectory with some colormap
            if keep_history:
                if force_2d:
                    ax.scatter([x[0] for x in pred_data][:t], [x[1] for x in pred_data][:t],
                               c=range(t),
                               cmap=self.colors[pid],
                               s=0.5)
                else:
                    ax.scatter([x[0] for x in pred_data][:t], [x[1] for x in pred_data][:t], self.round.timestamps[:t],
                               c=range(t),
                               cmap=self.colors[pid],
                               s=0.5)

            x = pred_data[t][0]
            y = pred_data[t][1]

            if force_2d:
                ax.scatter(x, y, s=100, marker='x', color=self.colors[pid][:-1], label=labels[pid] if labels is not None else None)

                if show_pred_vel_vector:
                    vel_vector_norm = np.linalg.norm(preds_data_vel[pid][t])
                    ax.quiver(pred_data[t][0], pred_data[t][1], preds_data_vel[pid][t][0] / vel_vector_norm,
                              preds_data_vel[pid][t][1] / vel_vector_norm,
                              color=self.colors[pid][:-1], linestyle='--')

                    perpendicular_vector = get_perpendicular_vector(np.array([preds_data_vel[pid][t][0], preds_data_vel[pid][t][1]]) / vel_vector_norm) * 10 + pred_data[t]
                    ax.plot((perpendicular_vector[0][0], pred_data[t][0], perpendicular_vector[1][0]),
                            (perpendicular_vector[0][1], pred_data[t][1], perpendicular_vector[1][1]), color='k',
                            linestyle='--')
            else:
                ax.scatter(x, y, z, s=100, marker='x', color=self.colors[pid][:-1], label=labels[pid] if labels is not None else None)

                if show_pred_vel_vector:
                    vel_vector_norm = np.linalg.norm(preds_data_vel[pid][t]) / 4
                    ax.quiver(X=pred_data[t][0], Y=pred_data[t][1], Z=z, U=preds_data_vel[pid][t][0] / vel_vector_norm,
                              V=preds_data_vel[pid][t][1] / vel_vector_norm, W=0.,
                              color=self.colors[pid][:-1], linestyle='--')

                    perpendicular_vector = get_perpendicular_vector(np.array([preds_data_vel[pid][t][0], preds_data_vel[pid][t][1]]) / vel_vector_norm) * 10 + pred_data[t]
                    ax.plot((perpendicular_vector[0][0], pred_data[t][0], perpendicular_vector[1][0]),
                            (perpendicular_vector[0][1], pred_data[t][1], perpendicular_vector[1][1]), (z, z, z),
                            color='k', linestyle='--')

        if labels is not None:
            ax.legend()

    def update_trajectories(self, agents_data: List[List[Tuple[float, float]]], preds_data: List[List[Tuple[float, float]]],
                            ax: plt.Axes, t: int, z: float, pred_labels: List[str] = None, keep_pred_history: bool = True,
                            show_arena_borders: bool = True, show_com: bool = False, agent_com=NDArray[float],
                            preds_data_vel: List[NDArray[float]] = None, show_pred_vel_vector: bool = False,
                            force_2d: bool = False, keep_com_history: bool = True, keep_agent_history: bool = False) -> None:
        if not keep_pred_history:
            force_2d = True

        if show_arena_borders:
            # Arena borders
            self.update_arena_borders(ax=ax, z=z, force_2d=force_2d)

        # predator data
        self.update_predator_trajectories(preds_data=preds_data, ax=ax, t=t, z=z, labels=pred_labels, keep_history=keep_pred_history,
                                          show_pred_vel_vector=show_pred_vel_vector, preds_data_vel=preds_data_vel, force_2d=force_2d)

        # agent data
        self.update_agent_trajectories(agents_data=agents_data, ax=ax, t=t, z=z, show_com=show_com, agent_com=agent_com,
                                       force_2d=force_2d, keep_com_history=keep_com_history, keep_agent_history=keep_agent_history)

        self.update_trajectories_ax_specs(ax, t=t, force_2d=force_2d)

    def update_metric_ax_specs(self, data_list: List[NDArray[float]], ax: plt.Axes,
                               t: int, t_start: int, time_window_dur: float,
                               metric: str, max_abs_val: float = None, keep_y_axis_stable: bool = False):

        if keep_y_axis_stable:
            ax.set_ylim(np.nanmin([np.nanmin(data) for data in data_list]),
                        np.nanmax([np.nanmax(data) for data in data_list]))
        else:
            if max_abs_val is None:
                max_abs_val = np.nanmax([np.nanmax(abs(data)) for data in data_list])
            ax.set_ylim(np.nanmax([np.nanmin([np.nanmin(data[t_start:t + 1]) for data in data_list]), -max_abs_val]),
                        np.nanmin([np.nanmax([np.nanmax(data[t_start:t + 1]) for data in data_list]), max_abs_val]))
        ax.set_xlim(self.round.timestamps[t] - 2 * time_window_dur / 3,
                    self.round.timestamps[t] + time_window_dur / 3)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(metric)
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
                     show_com: bool = False, show_pred_vel_vector: bool = False,
                     save: bool = False, out_file_path: str = "metrics.mp4") -> None:
        def update(t: int, args_dict: Dict) -> Dict:
            t_start = max([0, find_nearest(self.round.timestamps, self.round.timestamps[t] - 2*time_window_dur / 3)])
            z = self.round.timestamps[t]

            args_dict['ax'].cla()
            args_dict['ax_spe'].cla()
            args_dict['ax_acc'].cla()
            args_dict['ax_com'].cla()
            args_dict['ax_bor'].cla()
            args_dict['ax_att'].cla()
            args_dict['ax_npb'].cla()

            self.update_trajectories(agents_data=self.round.agents_data, preds_data=self.round.preds_data,
                                     preds_data_vel=preds_data_vel, show_pred_vel_vector=show_pred_vel_vector,
                                     ax=args_dict['ax'], t=t, z=z, show_com=show_com, agent_com=self.round.agent_com)

            # Speed axes
            self.update_predator_lines(preds_data_spe, ax=args_dict['ax_spe'], lines=[args_dict[f'vel_{pid}'] for pid in range(self.round.n_preds)],
                                       t=t, t_start=t_start)
            self.update_metric_ax_specs(preds_data_spe, ax=args_dict['ax_spe'],
                                        t=t, t_start=t_start, time_window_dur=time_window_dur,
                                        metric="speed [m/s]", max_abs_val=max_abs_speed)
            # Acceleration axes
            self.update_predator_lines(preds_data_acc_smoothed, ax=args_dict['ax_acc'],
                                       lines=[args_dict[f'acc_smoothed_{pid}'] for pid in range(self.round.n_preds)],
                                       t=t, t_start=t_start)
            self.update_predator_lines(preds_data_acc, ax=args_dict['ax_acc'],
                                       lines=[args_dict[f'acc_{pid}'] for pid in range(self.round.n_preds)],
                                       t=t, t_start=t_start, linestyle='--', alpha=0.6)
            self.update_metric_ax_specs(vcombine_lists_of_arrays([preds_data_acc, preds_data_acc_smoothed]), ax=args_dict['ax_acc'],
                                        t=t, t_start=t_start, time_window_dur=time_window_dur,
                                        metric="acceleration [m/s$^2$]", max_abs_val=max_abs_acceleration)
            # Predator distance to agents center of mass axes
            self.update_predator_lines(preds_data_com, ax=args_dict['ax_com'],
                                       lines=[args_dict[f'com_{pid}'] for pid in range(self.round.n_preds)],
                                       t=t, t_start=t_start)
            self.update_metric_ax_specs(preds_data_com, ax=args_dict['ax_com'],
                                        t=t, t_start=t_start, time_window_dur=time_window_dur,
                                        metric="distance from agents center of mass [m]")
            # Predator distance to border axes
            self.update_predator_lines(preds_data_bor, ax=args_dict['ax_bor'],
                                       lines=[args_dict[f'bor_{pid}'] for pid in range(self.round.n_preds)],
                                       t=t, t_start=t_start)
            self.update_metric_ax_specs(preds_data_bor, ax=args_dict['ax_bor'],
                                        t=t, t_start=t_start, time_window_dur=time_window_dur,
                                        metric="distance from closest border [m]")
            # Predator attack angle
            self.update_predator_lines(preds_data_att_smoothed, ax=args_dict['ax_att'],
                                       lines=[args_dict[f'att_smoothed_{pid}'] for pid in range(self.round.n_preds)],
                                       t=t, t_start=t_start)
            self.update_predator_lines(preds_data_att, ax=args_dict['ax_att'],
                                       lines=[args_dict[f'att_{pid}'] for pid in range(self.round.n_preds)],
                                       t=t, t_start=t_start, linestyle='--', alpha=0.6)
            self.update_metric_ax_specs(preds_data_att, ax=args_dict['ax_att'],
                                        t=t, t_start=t_start, time_window_dur=time_window_dur,
                                        metric="attack angle [degrees]", keep_y_axis_stable=True)
            # Number of prey agents behind predator
            self.update_predator_lines(preds_data_npb, ax=args_dict['ax_npb'],
                                       lines=[args_dict[f'npb_{pid}'] for pid in range(self.round.n_preds)],
                                       t=t, t_start=t_start)
            self.update_metric_ax_specs(preds_data_npb, ax=args_dict['ax_npb'],
                                        t=t, t_start=t_start, time_window_dur=time_window_dur,
                                        metric="#agents behind predator")

            return args_dict

        fig = plt.figure(figsize=(15.12, 9.82), dpi=100)
        gs = GridSpec.GridSpec(6, 2)

        ## Trajectory figure
        ax = fig.add_subplot(gs[:, 0], projection='3d')

        ## Metric figures
        if smoothing_args is None:
            smoothing_args = {'smoothing_method': 'window',
                              'kernel': lambda z: 1,
                              'window_size': 40}
        # Compute metrics
        preds_data_vel = self.round.compute_velocity(self.round.pred_data_arrs)
        preds_data_spe = self.round.compute_predator_speed()
        preds_data_acc_smoothed = self.round.compute_predator_acceleration(smoothing_args=smoothing_args)
        preds_data_acc = self.round.compute_predator_acceleration(smooth=False)
        preds_data_com = self.round.compute_predator_distance_to_agent_com()
        preds_data_bor = self.round.compute_predator_distance_to_border()
        preds_data_att_smoothed = self.round.compute_predator_attack_angle(smooth=True, smoothing_args=smoothing_args)
        preds_data_att = self.round.compute_predator_attack_angle()
        preds_data_npb = self.round.compute_n_preys_behind_predator()

        # speed
        ax_spe = fig.add_subplot(gs[0, 1])
        # acceleration
        ax_acc = fig.add_subplot(gs[1, 1])
        # predator distance to agents center of mass
        ax_com = fig.add_subplot(gs[2, 1])
        # predator distance from closest border
        ax_bor = fig.add_subplot(gs[3, 1])
        # predator attack angle
        ax_att = fig.add_subplot(gs[4, 1])
        # number of prey agents behind predator
        ax_npb = fig.add_subplot(gs[5, 1])

        args_dict = {'ax': ax,
                     'ax_spe': ax_spe,
                     'ax_acc': ax_acc,
                     'ax_com': ax_com,
                     'ax_bor': ax_bor,
                     'ax_att': ax_att,
                     'ax_npb': ax_npb}

        for pid in range(self.round.n_preds):
            args_dict[f'vel_{pid}'], = ax_spe.plot([], [], color=self.colors[pid][:-1])
            args_dict[f'acc_{pid}'], = ax_acc.plot([], [], color=self.colors[pid][:-1])
            args_dict[f'acc_smoothed_{pid}'], = ax_acc.plot([], [], color=self.colors[pid][:-1])
            args_dict[f'com_{pid}'], = ax_com.plot([], [], color=self.colors[pid][:-1])
            args_dict[f'bor_{pid}'], = ax_bor.plot([], [], color=self.colors[pid][:-1])
            args_dict[f'att_smoothed_{pid}'], = ax_att.plot([], [], color=self.colors[pid][:-1])
            args_dict[f'att_{pid}'], = ax_att.plot([], [], color=self.colors[pid][:-1])
            args_dict[f'npb_{pid}'], = ax_npb.plot([], [], color=self.colors[pid][:-1])

        ani = animation.FuncAnimation(fig=fig, func=update, frames=len(self.round.timestamps), interval=1, fargs=(args_dict,))

        plt.tight_layout()

        if save:
            print("Saving...")
            # saving to m4 using ffmpeg writer
            ani.save(out_file_path)

        plt.show()

        if save:
            plt.close()

    def plot_predator_acc_smoothings(self, time_window_dur: float = 2000., window_size: int = 40, show_com: bool = True,
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

            self.update_trajectories(agents_data=self.round.agents_data, preds_data=self.round.preds_data,
                                     ax=args_dict['ax'], t=t, z=z, show_com=show_com, agent_com=self.round.agent_com)

            # Raw predator acceleration
            self.update_predator_lines(preds_data_acc, ax=args_dict['ax_acc'],
                                       lines=[args_dict[f'acc_{pid}'] for pid in range(self.round.n_preds)],
                                       t=t, t_start=t_start)
            self.update_metric_ax_specs(preds_data_acc, ax=args_dict['ax_acc'],
                                        t=t, t_start=t_start, time_window_dur=time_window_dur,
                                        metric=f"acceleration (raw) [m/s$^2$]")
            # Predator acceleration with average smoothing kernel
            self.update_predator_lines(preds_data_acc_w, ax=args_dict['ax_acc_w'],
                                           lines=[args_dict[f'acc_w_{pid}'] for pid in range(self.round.n_preds)],
                                           t=t, t_start=t_start)
            self.update_predator_lines(preds_data_acc, ax=args_dict['ax_acc_w'],
                                       lines=[args_dict[f'acc_w_raw_{pid}'] for pid in range(self.round.n_preds)],
                                       t=t, t_start=t_start, linestyle='--', alpha=0.6)
            self.update_metric_ax_specs(vcombine_lists_of_arrays([preds_data_acc, preds_data_acc_w]), ax=args_dict['ax_acc_w'],
                                        t=t, t_start=t_start, time_window_dur=time_window_dur,
                                        metric=f"acceleration (average) [m/s$^2$]")
            # Predator acceleration with gaussian smoothing kernel
            self.update_predator_lines(preds_data_acc_w_gaus, ax=args_dict['ax_acc_w_gaus'],
                                           lines=[args_dict[f'acc_w_gaus_{pid}'] for pid in range(self.round.n_preds)],
                                           t=t, t_start=t_start)
            self.update_predator_lines(preds_data_acc, ax=args_dict['ax_acc_w_gaus'],
                                       lines=[args_dict[f'acc_w_gaus_raw_{pid}'] for pid in range(self.round.n_preds)],
                                       t=t, t_start=t_start, linestyle='--', alpha=0.6)
            self.update_metric_ax_specs(vcombine_lists_of_arrays([preds_data_acc, preds_data_acc_w_gaus]), ax=args_dict['ax_acc_w_gaus'],
                                        t=t, t_start=t_start, time_window_dur=time_window_dur,
                                        metric=f"acceleration (gaussian) [m/s$^2$]")
            # Predator acceleration with butterworth filter
            self.update_predator_lines(preds_data_acc_bw, ax=args_dict['ax_acc_bw'],
                                           lines=[args_dict[f'acc_bw_{pid}'] for pid in range(self.round.n_preds)],
                                           t=t, t_start=t_start)
            self.update_predator_lines(preds_data_acc, ax=args_dict['ax_acc_bw'],
                                       lines=[args_dict[f'acc_bw_raw_{pid}'] for pid in range(self.round.n_preds)],
                                       t=t, t_start=t_start, linestyle='--', alpha=0.6)
            self.update_metric_ax_specs(vcombine_lists_of_arrays([preds_data_acc, preds_data_acc_bw]), ax=args_dict['ax_acc_bw'],
                                        t=t, t_start=t_start, time_window_dur=time_window_dur,
                                        metric=f"acceleration \n (butterworth) [m/s$^2$]")
            # dts
            self.update_predator_lines([np.concatenate([np.array([0]), dts])], ax=args_dict['ax_dts'],
                                      lines=[args_dict['dts']],
                                       t=t, t_start=t_start, colors=['Blacks'])
            self.update_metric_ax_specs([np.concatenate([np.array([0]), dts])], ax=args_dict['ax_dts'],
                                        t=t, t_start=t_start, time_window_dur=time_window_dur,
                                        metric="dt [s]")

            return args_dict

        fig = plt.figure(figsize=(15.12, 9.82), dpi=100)
        gs = GridSpec.GridSpec(5, 2)

        dts = np.diff(self.round.timestamps)

        ## Trajectory figure
        ax = fig.add_subplot(gs[:, 0], projection='3d')

        ## Metric figures
        # Compute metrics
        preds_data_acc = self.round.compute_predator_acceleration(smooth=False)
        preds_data_acc_w = self.round.compute_predator_acceleration(smooth=True,
                                                                    smoothing_args={'smoothing_method': 'window',
                                                                                    'kernel': lambda z: 1,
                                                                                    'window_size': window_size})
        preds_data_acc_w_gaus = self.round.compute_predator_acceleration(smooth=True,
                                                                         smoothing_args={'smoothing_method': 'window',
                                                                                         'kernel': gaussian,
                                                                                         'window_size': window_size})
        preds_data_acc_bw = self.round.compute_predator_acceleration(smooth=True,
                                                                     smoothing_args={'smoothing_method': 'butterworth',
                                                                                     'fs': self.round.avg_fs/2,
                                                                                     'bw_fstart': 5,
                                                                                     'bw_fstop': 10,
                                                                                     'bw_order': 3})

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

        if save:
            print("Saving...")
            # saving to m4 using ffmpeg writer
            ani.save(out_file_path)
            plt.close()

        plt.tight_layout()
        plt.show()

    def plot_bout_trajectories(self, show_com: bool = True, discarded: bool = False, filter_number: int = 0,
                               save: bool = False, out_file_path: str = "bout_trajectories.mp4"):
        def update(t: int, args_dict: Dict) -> Dict:
            z = self.round.timestamps[t]

            args_dict['ax'].cla()

            self.update_trajectories(agents_data=agents_data,
                                     preds_data=preds_data,
                                     ax=args_dict['ax'], t=t, z=z,
                                     show_com=show_com, agent_com=agent_com)

            return args_dict

        fig = plt.figure(dpi=100)
        gs = GridSpec.GridSpec(1, 1)

        ## Trajectory figure
        ax = fig.add_subplot(gs[0, 0], projection='3d')

        args_dict = {'ax': ax}
        agents_data = self.round.agent_data_arrs_bouts_discarded[filter_number] if discarded else self.round.agent_data_arrs_bouts
        agent_com = self.round.agent_com_bouts_discarded[filter_number] if discarded else self.round.agent_com_bouts
        preds_data = self.round.pred_data_arrs_bouts_discarded[filter_number] if discarded else self.round.pred_data_arrs_bouts
        timestamps = self.round.timestamps_bouts_discarded[filter_number] if discarded else self.round.timestamps_bouts

        ani = animation.FuncAnimation(fig=fig, func=update, frames=len(timestamps), interval=1, fargs=(args_dict,))

        if save:
            print("Saving...")
            # saving to m4 using ffmpeg writer
            ani.save(out_file_path)
            plt.close()

        plt.tight_layout()
        plt.show()

    def plot_bout_division_hbars(self, t: int = None, ax: plt.Axes = None, time_window_dur: float = None, set_y_labels: bool = True, separate_predators: bool = True):
        y_pos = ['Predator 1', 'Predator 2', 'Predator 3'] if set_y_labels else np.arange(3) + 1 if separate_predators else np.full(3, 1)

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        min_bout_start = self.round.timestamps[-1]
        max_bout_end = self.round.timestamps[0]
        for pid in range(self.round.n_preds):
            left = self.round.timestamps[0]
            for i, (bout_start, bout_end) in enumerate(self.round.pred_bout_bounds_filtered[pid]):
                bout_id = self.round.pred_bout_ids_filtered[pid][i]

                if time_window_dur is not None:
                    if self.round.timestamps[bout_start] >= self.round.timestamps[t] + time_window_dur / 3:
                        break
                    #elif self.round.timestamps[bout_end] > self.round.timestamps[t] + time_window_dur / 3:
                    #    bout_end = find_nearest(self.round.timestamps, self.round.timestamps[t] + time_window_dur / 3)

                if separate_predators:
                    ax.barh(y_pos[pid], width=self.round.timestamps[bout_start] - left, left=left, color='grey')
                else:
                    min_bout_start = np.min([min_bout_start, self.round.timestamps[bout_start]])
                    max_bout_end = np.max([max_bout_end, self.round.timestamps[bout_end]])
                left = self.round.timestamps[bout_end]

                if time_window_dur is None or (time_window_dur is not None and self.round.timestamps[bout_start] >= self.round.timestamps[t] - 2 * time_window_dur / 3):
                    ax.text(x=self.round.timestamps[bout_start]+(self.round.timestamps[bout_end] - self.round.timestamps[bout_start])/2, y=(-0.85*pid if time_window_dur is None else 0) + (self.round.n_preds + 0.71 if separate_predators else 1.43), s=bout_id,
                            c=self.colors[pid][:-1], size=3 if time_window_dur is None else 8, horizontalalignment='center', verticalalignment='center', rotation_mode='anchor')
                ax.barh(y_pos[pid], width=self.round.timestamps[bout_end] - self.round.timestamps[bout_start], left=self.round.timestamps[bout_start], color=self.colors[pid][:-1], alpha=0.6 if not separate_predators else 1)
            if separate_predators:
                ax.barh(y_pos[pid], width=self.round.timestamps[-1] - left, left=left, color='grey')
        if not separate_predators:
            ax.barh(y_pos[0], width=self.round.timestamps[-1] - max_bout_end, left=max_bout_end, color='white')
            ax.barh(y_pos[0], width=min_bout_start - self.round.timestamps[0], left=self.round.timestamps[0], color='white')
        ax.set_xlabel("Time [s]")
        ax.set_yticks(y_pos[:self.round.n_preds] if set_y_labels else [])
        ax.set_title("Predator bouts (colored) timeline")

        if time_window_dur is not None:
            ax.set_xlim(xmin=self.round.timestamps[t] - 2 * time_window_dur / 3,
                        xmax=self.round.timestamps[t] + time_window_dur / 3)

        if ax is None:
            plt.show()

    def plot_bout_division(self, show_com: bool = False, separate_predators: bool = False, fountain_metric_method: str = "convexhull",
                           keep_pred_history: bool = False, keep_com_history: bool = False, keep_agent_history: bool = False,
                           show_pred_vel_vector: bool = True, time_window_dur: float = None, show_n_agents_behind: bool = True,
                           save: bool = False, out_file_path: str = "bout_divisions.mp4", show: bool = True) -> None:
        def update(t: int, args_dict: Dict) -> Dict:
            t_start = max([0, find_nearest(self.round.timestamps, self.round.timestamps[t] - 2*time_window_dur / 3)])

            args_dict['ax'].cla()
            args_dict['ax_bout_div'].cla()
            if show_n_agents_behind:
                args_dict['ax_npb'].cla()

            if t == 0:
                args_dict['in_bout'] = [t in pred_bout_starts[pid] for pid in range(self.round.n_preds)]

            for pid in range(self.round.n_preds):
                if args_dict['in_bout'][pid]:
                    args_dict['in_bout'][pid] = args_dict['in_bout'][pid] and (t not in pred_bout_ends[pid])
                else:
                    args_dict['in_bout'][pid] = t in pred_bout_starts[pid]

            self.update_trajectories(agents_data=self.round.agents_data, preds_data=self.round.preds_data, pred_labels=[".", "."],
                                     ax=args_dict['ax'], t=t, z=self.round.timestamps[t], show_com=show_com, agent_com=self.round.agent_com,
                                     show_pred_vel_vector=show_pred_vel_vector, preds_data_vel=preds_data_vel,
                                     keep_pred_history=keep_pred_history, keep_com_history=keep_com_history, keep_agent_history=keep_agent_history)

            pred_labels = []
            for pid in range(self.round.n_preds):
                for bout_id in range(len(self.round.bout_evasion_start_times[pid])):
                    bout_evasion_start_time = self.round.bout_evasion_start_times[pid][bout_id]
                    bout_evasion_end_time = self.round.bout_evasion_end_times[pid][bout_id]
                    if time_window_dur is None or (time_window_dur is not None and self.round.timestamps[t] - 2 * time_window_dur / 3 <= bout_evasion_start_time <= self.round.timestamps[t] + time_window_dur / 3):
                        if bout_evasion_start_time >= 0:
                            args_dict['ax_bout_div'].axvline(x=bout_evasion_start_time,
                                                             ymin=0.04 + pid * (separate_predators), ymax=(pid * separate_predators + 1) * 0.84,
                                                             color=self.colors[pid][:-1], linestyle="--", linewidth=1)
                            args_dict['ax_bout_div'].axvline(x=bout_evasion_end_time,
                                                             ymin=0.04 + pid * (separate_predators), ymax=(pid * separate_predators + 1) * 0.84,
                                                             color=self.colors[pid][:-1], linestyle="--", linewidth=1)
                            args_dict['ax_bout_div'].text(x=(bout_evasion_start_time + bout_evasion_end_time)/2, y=1 + pid * (separate_predators),
                                                          s=f"{bout_evasion_fountain_metric[pid][bout_id]:.3f}",
                                                          ha='center', va='top', rotation='vertical')

                if args_dict['in_bout'][pid]:
                    pred_labels.append("in bout")
                else:
                    pred_labels.append("")

                if time_window_dur is None:
                    args_dict['bout_indicator'][pid] = mpl.patches.Ellipse((50 + pid*50, self.round.n_preds + 0.75 if separate_predators else 1.5),
                                                                            30, height=0.05, color=self.colors[pid][:-1],
                                                                           clip_on=False, alpha=1. if args_dict['in_bout'][pid] else 0.4)
                    args_dict['ax_bout_div'].add_patch(args_dict['bout_indicator'][pid])

            args_dict['legend'] = args_dict['ax'].legend(loc='upper right')
            [args_dict['legend'].get_texts()[pid].set_text(pred_labels[pid]) for pid in range(self.round.n_preds)]

            # Bout divison axes
            self.plot_bout_division_hbars(t=t, ax=args_dict['ax_bout_div'], set_y_labels=False, separate_predators=separate_predators, time_window_dur=time_window_dur)
            # add time bar
            args_dict['ax_bout_div'].axvline(x=self.round.timestamps[t], ymin=-0.1, ymax=(self.round.n_preds if separate_predators else 1)*0.88)
            if time_window_dur is None:
                args_dict['ax_bout_div'].set_xlim([0, self.round.timestamps[-1]])

            if show_n_agents_behind:
                # Number of prey agents behind predator
                self.update_predator_lines(preds_data_npb, ax=args_dict['ax_npb'],
                                           lines=[args_dict[f'npb_{pid}'] for pid in range(self.round.n_preds)],
                                           t=t, t_start=t_start)
                self.update_metric_ax_specs(preds_data_npb, ax=args_dict['ax_npb'], keep_y_axis_stable=True,
                                            t=t, t_start=t_start, time_window_dur=time_window_dur,
                                            metric="#agents behind predator")

            return args_dict

        preds_data_vel = self.round.compute_predator_velocity()
        bout_evasion_fountain_metric = self.round.compute_bout_evasion_fountain_metric()

        fig = plt.figure(figsize=(15.12, 9.82), dpi=100)
        gs = GridSpec.GridSpec(2 if show_n_agents_behind else 1, 2)

        ## Trajectory figure
        ax = fig.add_subplot(gs[:, 0], projection='3d' if keep_pred_history else None)

        ax_bout_div = fig.add_subplot(gs[0, 1])

        args_dict = {'ax': ax, 'ax_bout_div': ax_bout_div,
                     'in_bout': [False for _ in range(self.round.n_preds)], 'bout_id': -1,
                     'legend': ax.legend(loc='upper right'), 'bout_indicator': [mpl.patches.Ellipse((50 + pid*50, 1.5), 30, height=0.05, color=self.colors[pid][:-1],
                                            clip_on=False, alpha=0.4) for pid in range(self.round.n_preds)]}

        if show_n_agents_behind:
            preds_data_npb = self.round.compute_n_preys_behind_predator()
            ax_npb = fig.add_subplot(gs[1, 1], sharex=ax_bout_div)
            args_dict['ax_npb'] = ax_npb

            for pid in range(self.round.n_preds):
                args_dict[f'npb_{pid}'], = ax_npb.plot([], [], color=self.colors[pid][:-1])
        args_dict['time_bar'] = args_dict['ax_bout_div'].axvline(self.round.timestamps[0])
        pred_bout_starts = [[self.round.pred_bout_bounds_filtered[pid][i][0] for i in range(len(self.round.pred_bout_bounds_filtered[pid]))] for pid in range(self.round.n_preds)]
        pred_bout_ends = [[self.round.pred_bout_bounds_filtered[pid][i][1] for i in range(len(self.round.pred_bout_bounds_filtered[pid]))] for pid in range(self.round.n_preds)]

        ani = animation.FuncAnimation(fig=fig, func=update, frames=len(self.round.timestamps), interval=1, fargs=(args_dict,))

        if save:
            print("Saving...")
            # saving to m4 using ffmpeg writer
            ani.save(out_file_path, fps=self.fps)

        if show:
            plt.show()

        if save:
            plt.close()

    def plot_single_bout(self, bout_id: str, show_com: bool = True, keep_com_history: bool = True, fountain_metric_method: str = "convexhull",
                         keep_pred_history: bool = True, keep_agent_history: bool = True, force_2d: bool = True,
                         show_pred_vel_vector: bool = True, mark_speed_spike: bool = False, speed_spike_threshold: float = 10.,
                         save: bool = False, out_file_path: str = "bout.mp4", show: bool = True):
        def update(t: int, args_dict: Dict) -> Dict:

            args_dict['ax'].cla()
            args_dict['ax_evasion_trajectory'].cla()
            args_dict['ax_bout_info'].cla()

            args_dict['ax_bout_info'].set_xlabel("Time [s]")
            args_dict['ax_bout_info'].set_ylabel("Predator in bout speed [m/s]")

            self.update_trajectories(agents_data=agents_data, preds_data=preds_data,
                                     ax=args_dict['ax'], t=t, z=timestamps[t], show_com=show_com, agent_com=agent_com,
                                     show_pred_vel_vector=show_pred_vel_vector, preds_data_vel=preds_data_vel, keep_pred_history=keep_pred_history,
                                     keep_com_history=keep_com_history, keep_agent_history=keep_agent_history, force_2d=force_2d)

            if bout_evasion_start_time >= 0:
                args_dict['ax_bout_info'].axvline(x=bout_evasion_start_time, ymin=0, ymax=np.max(pred_speed),
                                                  color=self.colors[pid_in_bout][:-1], linestyle="--", linewidth=1)
                args_dict['ax_bout_info'].axvline(x=bout_evasion_end_time, ymin=0, ymax=np.max(pred_speed),
                                                  color=self.colors[pid_in_bout][:-1], linestyle="--", linewidth=1)
                fontsize = "small"

                args_dict['ax_bout_info'].text(x=timestamps[0]+0.05, y=np.max(pred_speed)*1.05,
                                               s=f"Circularity metric: {bout_evasion_circularity:.3f}", fontsize=fontsize)
                args_dict['ax_bout_info'].text(x=timestamps[0]+0.05, y=np.max(pred_speed),
                                               s=f"M Convexity metric: {bout_evasion_convexity_m:.3f}", fontsize=fontsize)
                args_dict['ax_bout_info'].text(x=timestamps[0]+0.05, y=np.max(pred_speed)*0.95,
                                               s=f"E Convexity metric: {bout_evasion_convexity_e:.3f}", fontsize=fontsize)
                args_dict['ax_bout_info'].text(x=timestamps[0]+0.05, y=np.max(pred_speed)*0.9,
                                               s=f"M Polarisation metric: {bout_evasion_polarisation_m:.3f}", fontsize=fontsize)
                args_dict['ax_bout_info'].text(x=timestamps[0]+0.05, y=np.max(pred_speed)*0.85,
                                               s=f"E Polarisation metric: {bout_evasion_polarisation_e:.3f}", fontsize=fontsize)

                self.update_trajectories(agents_data=[agents_data[aid][bout_evasion_start_id:bout_evasion_end_id] for aid in range(self.round.n_agents)],
                                         preds_data=[preds_data[pid][bout_evasion_start_id:bout_evasion_end_id] for pid in range(self.round.n_preds)],
                                         ax=args_dict['ax_evasion_trajectory'], t=len(timestamps[bout_evasion_start_id:bout_evasion_end_id]) - 1, z=timestamps[bout_evasion_start_id:bout_evasion_end_id][-1],
                                         show_com=show_com, agent_com=agent_com[bout_evasion_start_id:bout_evasion_end_id],
                                         show_pred_vel_vector=False, keep_pred_history=True, show_arena_borders=True, force_2d=force_2d,
                                         keep_com_history=keep_com_history, keep_agent_history=True
                                         )
                if fountain_metric_method == "convexhull":
                    hull = self.round.preds_convex_hulls[pid_in_bout][bout_idx]
                    hull_points = self.round.preds_convex_hull_points[pid_in_bout][bout_idx]
                    #args_dict['ax_evasion_trajectory'].plot(hull_points[hull.vertices,0], hull_points[hull.vertices,1], 'r--', lw=2)
                    for simplex in hull.simplices:
                        args_dict['ax_evasion_trajectory'].plot(hull_points[simplex, 0], hull_points[simplex, 1], 'k-')

                    bounding_circle_x, bounding_circle_y = self.round.preds_bounding_circles[pid_in_bout][bout_idx].xy
                    args_dict['ax_evasion_trajectory'].plot(bounding_circle_x, bounding_circle_y)

            if mark_speed_spike:
                args_dict['ax_bout_info'].text(x=(timestamps[0] + timestamps[-1])/2, y=np.max(pred_speed)*1.05,
                                               s=f"Speed spike present: {speed_spike_present}")

            # add time bar
            args_dict['ax_bout_info'].axvline(x=timestamps[t], ymin=-0.1, ymax=np.max(pred_speed)*0.98)
            args_dict['ax_bout_info'].set_xlim([timestamps[0], timestamps[-1]])
            #add speed
            args_dict['ax_bout_info'].set_ylim([0, np.max(pred_speed)*1.1])
            args_dict['ax_bout_info'].plot(timestamps, pred_speed, color=self.colors[pid_in_bout][:-1])
            #args_dict['ax_bout_info'].plot(timestamps, prey_on_both_sides_of_pred[pid_in_bout], color=self.colors[pid_in_bout][:-1])

            return args_dict

        pid_in_bout = int(bout_id.split('_')[1]) - 1 if self.round.n_preds > 1 else 0
        bout_idx = np.argwhere(self.round.pred_bout_ids_filtered[pid_in_bout] == bout_id)[0][0]
        bout_start, bout_end = self.round.pred_bout_bounds[pid_in_bout][np.argwhere(self.round.pred_bout_ids[pid_in_bout] == bout_id)[0][0]]

        agents_data = [self.round.agents_data[aid][bout_start:bout_end] for aid in range(self.round.n_agents)]
        agent_com = self.round.agent_com[bout_start:bout_end]
        preds_data = [self.round.preds_data[pid][bout_start:bout_end] for pid in range(self.round.n_preds)]
        pred_speed = self.round.compute_predator_speed()
        pred_speed = pred_speed[pid_in_bout][bout_start:bout_end]
        timestamps = self.round.timestamps[bout_start:bout_end]

        preds_data_vel = [self.round.compute_predator_velocity()[pid][bout_start:bout_end] for pid in range(self.round.n_preds)]
        preds_data_speed = [self.round.compute_predator_speed()[pid][bout_start:bout_end] for pid in range(self.round.n_preds)]
        speed_spike_present = np.any(preds_data_speed[pid_in_bout] > speed_spike_threshold)
        #prey_on_both_sides_of_pred = [self.round.check_if_preys_on_both_sides_of_predator()[pid][bout_start:bout_end] for pid in range(self.round.n_preds)]

        bout_evasion_start_time = self.round.bout_evasion_start_times[pid_in_bout][bout_idx]
        bout_evasion_end_time = self.round.bout_evasion_end_times[pid_in_bout][bout_idx]
        bout_evasion_start_id = self.round.bout_evasion_start_ids[pid_in_bout][bout_idx]
        bout_evasion_end_id = self.round.bout_evasion_end_ids[pid_in_bout][bout_idx]
        bout_evasion_circularity = self.round.compute_bout_evasion_circularity()[pid_in_bout][bout_idx]
        bout_evasion_convexity_m = self.round.compute_bout_evasion_convexity(compute_at="middle")[pid_in_bout][bout_idx]
        bout_evasion_convexity_e = self.round.compute_bout_evasion_convexity(compute_at="end")[pid_in_bout][bout_idx]
        bout_evasion_polarisation_m = self.round.compute_bout_evasion_polarisation(compute_at="middle")[pid_in_bout][bout_idx]
        bout_evasion_polarisation_e = self.round.compute_bout_evasion_polarisation(compute_at="end")[pid_in_bout][bout_idx]

        fig = plt.figure(figsize=(15.12, 9.82), dpi=100)
        fig.suptitle(f"Experiment ID: {self.round.file_path.split('/')[-3]}, Bout ID: {bout_id}")
        gs = GridSpec.GridSpec(2, 2)

        ## Trajectory figure
        ax = fig.add_subplot(gs[:, 0], projection='3d' if not force_2d else None)

        ax_evasion_trajectory = fig.add_subplot(gs[0, 1])
        ax_bout_info = fig.add_subplot(gs[1, 1])

        args_dict = {'ax': ax, 'ax_evasion_trajectory': ax_evasion_trajectory, 'ax_bout_info': ax_bout_info}

        args_dict['time_bar'] = args_dict['ax_bout_info'].axvline(timestamps[0])

        ani = animation.FuncAnimation(fig=fig, func=update, frames=len(timestamps), interval=1, fargs=(args_dict,))

        if save:
            print("Saving...")
            # saving to m4 using ffmpeg writer
            ani.save(out_file_path, fps=self.fps)

        if show:
            plt.show()

        if save:
            plt.close()

    def plot_bout_metric(self, metric: Iterable[Iterable[float]],
                         reference_times: Iterable[Iterable[float]], reference_ids: Iterable[Iterable[float]],
                         overlay_bouts: bool = False, metric_name: str = 'metric', reference_time_name: str = 'reference point'):
        fig = plt.figure()
        gs = GridSpec.GridSpec(self.round.n_preds, 1)

        for pid in range(self.round.n_preds):
            ax = fig.add_subplot(gs[pid, :], sharex=ax if pid > 0 and overlay_bouts else None)
            first = True
            count = 0
            min_reference_id = np.min(np.array(reference_ids[pid])[np.array(reference_ids[pid]) >= 0])
            for i, (bout_start, bout_end) in enumerate(self.round.pred_bout_bounds_filtered[pid]):
                reference_time = reference_times[pid][i]
                if reference_time >= 0:
                    count += 1
                    if first:
                        mean_metric_relative_time = self.round.timestamps[bout_start:bout_end] - reference_time
                        mean_metric = metric[pid][bout_start:bout_end][reference_ids[pid][i] - min_reference_id:reference_ids[pid][i]+1]
                        last_bout_end = self.round.timestamps[bout_start]
                        first = False
                    else:
                        mean_metric_relative_time = self.round.timestamps[bout_start:bout_end] - reference_time if mean_metric_relative_time[0] < (self.round.timestamps[bout_start:bout_end] - reference_time)[0] else mean_metric_relative_time
                        mean_metric += metric[pid][bout_start:bout_end][reference_ids[pid][i] - min_reference_id:reference_ids[pid][i]+1]

                    alpha = ((int(self.round.pred_bout_ids_filtered[pid][i].split('_')[0])+1)/len(self.round.pred_bout_bounds[pid]))*0.9 + (1-0.9)
                    ax.plot(self.round.timestamps[bout_start:bout_end] - reference_time * overlay_bouts - (self.round.timestamps[bout_start] - last_bout_end - 3) * (not overlay_bouts), metric[pid][bout_start:bout_end],
                            color=self.colors[pid][:-1], alpha=alpha)
                    if not overlay_bouts:
                        ax.scatter(reference_time - (self.round.timestamps[bout_start] - last_bout_end - 3), metric[pid][bout_start:bout_end][self.round.timestamps[bout_start:bout_end] == reference_time],
                                   color=self.colors[pid][:-1], alpha=alpha)
                        ax.axvline(reference_time - (self.round.timestamps[bout_start] - last_bout_end - 3), color='k', linestyle='--')
                        ax.text(reference_time - (self.round.timestamps[bout_start] - last_bout_end - 3), -.05, self.round.pred_bout_ids_filtered[pid][i],
                                color=self.colors[pid][:-1], transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=7.8)
                    else:
                        ax.scatter(0, metric[pid][bout_start:bout_end][self.round.timestamps[bout_start:bout_end] == reference_time],
                                   color=self.colors[pid][:-1], alpha=alpha)

                    last_bout_end = self.round.timestamps[bout_end] - (self.round.timestamps[bout_start] - last_bout_end - 3)
            mean_metric /= count
            if overlay_bouts:
                ax.plot(mean_metric_relative_time[mean_metric_relative_time <= 0], mean_metric, color='black', linestyle='-.')
                ax.axvline(0, color='k', linestyle='--')
                ax.set_xlabel(f"Time relative to {reference_time_name}")
            else:
                ax.set_xticks([])

            ax.set_ylabel(metric_name)

        plt.tight_layout()
        plt.show()

    def plot_bout_metric_static(self, metric: Iterable[Iterable[float]], metric_name: str = 'metric', overlay_bouts: bool = False):
        fig = plt.figure()
        gs = GridSpec.GridSpec(self.round.n_preds, 1)

        for pid in range(self.round.n_preds):
            ax = fig.add_subplot(gs[pid, :], sharex=ax if pid > 0 and overlay_bouts else None)

            for bout_id in range(len(self.round.pred_bout_bounds_filtered[pid])):
                if metric[pid][bout_id] >= 0:

                    alpha = ((int(self.round.pred_bout_ids_filtered[pid][bout_id].split('_')[0])+1)/len(self.round.pred_bout_bounds[pid]))*0.9 + (1-0.9)
                    ax.plot(0 + bout_id * (not overlay_bouts), metric[pid][bout_id],
                            color=self.colors[pid][:-1], alpha=alpha)
                    if not overlay_bouts:
                        ax.scatter(bout_id, metric[pid][bout_id], color=self.colors[pid][:-1], alpha=alpha)
                    else:
                        ax.scatter(0, metric[pid][bout_id], color=self.colors[pid][:-1], alpha=alpha)

            ax.set_xticks([])
            ax.set_ylabel(metric_name)

        plt.tight_layout()
        plt.show()



