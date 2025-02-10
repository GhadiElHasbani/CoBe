import numpy as np
from typing import List
from numpy.typing import NDArray


def limit_norm(x, limit) -> NDArray:
    """Limits the norm of n-dimensional vector
    Args:
        x (ndarray): Vector
        limit (float): Limit norm
    Returns:
        x (ndarray): Vector with norm limited to limit
    """
    norm = np.linalg.norm(x)
    if norm > limit:
        x /= norm  # make x unit vector
        x *= limit  # scale x by limit
    return x


def flee_cmd(pos_pred, vel_pred, pos_prey, flee_ang, flee_str, flee_range):
    command_flee_pred = [0, 0]

    pos_pred = pos_pred - pos_prey
    vel_pred = vel_pred / 2
    dist_pred = abs(np.linalg.norm(pos_pred))
    pred_hunt = True  # disable for orthog. vector

    ort_pv = [0, 0]
    ort_pv[0] = -vel_pred[1]
    ort_pv[1] = vel_pred[0]

    command_flee = np.zeros(2)

    side = 1  # if 1-> prey is right, if -1-> prey is left (?)

    ### Calc relative distance vector and corresponding unit vector
    r_ip = pos_pred
    ### Calc if prey is left or right from pred
    ### (project r_ip on unit vector right from u of pred  -> [u[1], -u[0]])
    if (r_ip[0] * vel_pred[1] - r_ip[1] * vel_pred[0]) > 0:
        side = -1

    fleeang = np.radians(flee_ang)

    ang = np.atan2(r_ip[1], r_ip[0]) - side * fleeang

    ang = np.fmod(ang, 2 * np.pi)
    ang = np.fmod(2 * np.pi + ang, 2 * np.pi)

    x0 = flee_range
    if dist_pred > 0:  # changed this from 0, makes more sense
        command_flee[0] = np.cos(ang)
        command_flee[1] = np.sin(ang)

    steepness = -1

    fleestr = 0

    if dist_pred < x0:
        fleestr = 0.5 * (np.tanh(steepness * (dist_pred - x0)) + 1.0)
    command_flee_pred = command_flee_pred - command_flee * fleestr

    if abs(command_flee_pred[0]) + abs(command_flee_pred[1]) != 0:
        command_flee_pred = command_flee_pred / abs(np.linalg.norm(command_flee_pred))

    return command_flee_pred * flee_str


def addForces(flock, flee, vel_self, dt, dphi):
    phi = np.arctan2(vel_self[1], vel_self[0])

    force = (flock + flee)

    phi_f = -force[0] * np.sin(phi) + force[1] * np.cos(phi)

    noisep = np.sqrt(dt * 2 * dphi)
    noise = noisep * np.random.normal()

    if abs(vel_self[0]) + abs(vel_self[1]) != 0:

        vproj = np.sqrt(vel_self[0] * vel_self[0] + vel_self[1] * vel_self[1])
        phi += ((phi_f * dt + noise)) / vproj

        phi = np.fmod(phi, 2 * np.pi)
        phi = np.fmod(2 * np.pi + phi, 2 * np.pi)
    else:
        phi = np.atan2(force[1], force[0])

    return np.array((np.cos(phi), np.sin(phi)))


def extract_SII(vel_rep, pos_rep, pred_vel_rep, pred_pos_rep, flee,
                flee_ang=30, prey_amount=10, dt=0.02, flee_str=50,
                flee_range=10, dphi=0.2, traj_length=400) -> List[float]:
    first = []
    last = []

    time = len(vel_rep[:, 0, 0])

    for ii in range(len(flee[0, :])):

        flee_t = np.argwhere(flee[:, ii] > 0)

        if len(flee_t) == 0:
            first.append(len(flee[:, 0]) - 1)
        else:
            first.append(flee_t[0][0])

        flee_end = np.argwhere(flee[first[ii]:, ii] <= 40)

        if len(flee_end) == 0:
            last.append(len(flee[:, 0]) - 1)

        else:
            if 400 + first[ii] < time:
                last.append(traj_length + first[ii])
            else:
                last.append(time - 1)

    prey_targets_idx = np.argsort(np.array(first))[:prey_amount]

    SII = []

    for idx in prey_targets_idx:

        start = first[idx]
        end = last[idx]

        prey_run = np.copy(pos_rep[start:end, idx, :])

        t_steps = end - start

        pred = pred_pos_rep[start:end, :]
        pred_vel = pred_vel_rep[start:end, :]

        pos_prey = np.copy(pos_rep[start, idx, :])
        vel_prey = np.copy(vel_rep[start, idx, :])
        prey_save = np.zeros((t_steps, 2))

        for p in range(t_steps):
            pos_pred = np.array(pred[p, :])
            vel_pred = np.array(pred_vel[p, :])

            f_flee = flee_cmd(pos_pred, vel_pred, pos_prey, flee_ang, flee_str, flee_range)
            f_prey = addForces(np.array((0, 0)), f_flee, vel_prey, dt, dphi)

            vel_prey = limit_norm(f_prey / np.linalg.norm(f_prey), 1)

            pos_prey += dt * vel_prey
            prey_save[p, :] = pos_prey

        a = prey_save
        b = prey_run
        distances = np.sqrt(((a - b) ** 2).sum(axis=-1))
        SII.append(np.mean(distances))  # simple euc
    return SII
