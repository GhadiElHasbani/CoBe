from typing import List, Tuple, Callable, Iterable, Dict
import numpy as np
import pandas
from numpy._typing import ArrayLike
from numpy.typing import NDArray
from shapely.geometry import Polygon, Point
from scipy.signal import butter, lfilter
import os


def convert_list_of_tuples_to_array(list_of_tuples: List[Tuple]) -> NDArray:
    return np.vstack([np.array(element) for element in list_of_tuples])


def convert_array_to_list_of_tuples(array: NDArray) -> List[Tuple]:
    return list(map(tuple, array))


def gaussian(z: NDArray[float]) -> NDArray[float]:
    return np.exp(z**2/2)/np.sqrt(2*np.pi)


def kernel_smoother(ts: NDArray[float],
                    window_size: int,
                    kernel: Callable = lambda z: 1) -> NDArray[float]:

    smoothed_ts = np.zeros(len(ts))

    for i in range(len(ts)):
        start = max([0, i - int(window_size / 2)])
        end = min([i + int(window_size / 2), len(ts)])
        for j, ts_j in enumerate(ts[start:end]):
            w_ij = kernel((ts[i] - ts_j) / window_size) / np.sum([kernel((ts[i] - ts_k) / window_size) for ts_k in ts[start:end]])
            smoothed_ts[i] += w_ij * ts_j

    return smoothed_ts


def butterworth_smoother(ts: NDArray[float],
                         fs: float,
                         f_start: float,
                         f_stop: float,
                         order: int = 5):
    b, a = butter(order, [f_start, f_stop], fs=fs, btype='band')
    y = lfilter(b, a, ts)
    return y


def smooth_array(ts: NDArray[float],
                 smoothing_method: str = 'window',
                 window_size: int = 40,
                 stride: int = 1,
                 kernel: Callable = lambda z: 1,
                 fs: float = None,
                 bw_order: int = 5,
                 bw_fstart: float = None,
                 bw_fstop: float = None
                 ) -> NDArray[float]:
    if smoothing_method == 'window':
        print(f"smoothing with kernel {kernel}")
        smoothed_ts = kernel_smoother(ts, window_size, kernel)
    elif smoothing_method == 'butterworth':
        print(f"smoothing {fs}Hz data with {bw_order}-order butterworth filter [{bw_fstart}:{bw_fstop}]Hz")
        smoothed_ts = butterworth_smoother(ts, fs, bw_fstart, bw_fstop, bw_order)
    else:
        raise ValueError(f"Smoothing method '{smoothing_method}' not supported. Choose one of {['window', 'butterworth']}")

    return smoothed_ts[::stride]


def smooth_metric(data_arrs: List[NDArray[float]],
                  smoothing_args: Dict = None) -> List[NDArray[float]]:
    if smoothing_args is None:
        smoothing_args = {'smoothing_method': 'window',
                          'kernel': lambda z: 1,
                          'window_size': 40}

    data_arrs_smooth = [smooth_array(ts=data_arr, **smoothing_args) for data_arr in data_arrs]

    return data_arrs_smooth


def shortest_dist_to_polygon(vertices: List[Tuple[float, float]], points: NDArray[float]) -> NDArray[float]:
    poly = Polygon(vertices)
    dists = np.zeros(len(points))
    for i in range(len(points)):
        point = points[i]
        point_shape = Point(point[0], point[1])
        dists[i] = poly.exterior.distance(point_shape)

    return dists


def shortest_dist_to_circle(center: Tuple[float, float], radius: float, points: NDArray[float]) -> NDArray[float]:
    return ((((points[0] - center[0]) ** 2) + ((points[1] - center[1]) ** 2)) ** (1 / 2)) - radius


def find_nearest(array: ArrayLike, value: float) -> int:
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def euclidean_distance(points_a: NDArray[float], points_b: NDArray[float]) -> NDArray[float]:
    return np.sqrt(np.sum((points_a - points_b)**2, axis=0))


def get_bounds(arr: NDArray[bool], margin: int = 0) -> Tuple[List[Tuple[int, int]], List[int]]:
    bounds = []
    durations = []
    current_start = None

    def update_current_start(i: int, margin: int):
        return max([i - margin, 0])

    def update_current_end(i: int, margin: int, bounds: List, durations: List):
        current_end = min([len(arr) - 1, i + margin])
        if not current_start == 0 and not current_end == len(arr) - 1:
            bounds.append((current_start, current_end + 1))
            durations.append(current_end + 1 - current_start)

    for i in range(len(arr)):
        if arr[i]:
            if i == 0:
                current_start = update_current_start(i, margin)
            elif i == len(arr) - 1 and arr[i-1]:
                update_current_end(i, margin, bounds, durations)
            elif 0 < i < len(arr) - 1:
                if arr[i-1] and not arr[i+1]:
                    update_current_end(i, margin, bounds, durations)
                    current_start = None
                elif arr[i+1] and not arr[i-1]:
                    current_start = update_current_start(i, margin)

    return bounds, durations


def write_column_to_csv(file_path: str, column: Iterable, column_name: str):
    out_file = '/'.join(file_path.split('/')[:-1]) + file_path.split('/')[-1].split('.')[-2] + '_t.' + file_path.split('/')[-1].split('.')[-1]
    print(f"Writing timestamps to {out_file}")
    with open(file_path, 'r') as istr:
        with open(out_file, 'w') as ostr:
            for i, line in enumerate(istr):
                line = line.rstrip('\n') + f',{column_name},{column[i]}'
                print(line, file=ostr)
    os.rename(out_file, file_path)


def vcombine_lists_of_arrays(lists_of_arrays: Iterable[List[NDArray]]) -> List[NDArray]:
    list_of_arrays = [np.array([]) for arr in lists_of_arrays[0]]

    for i in range(len(list_of_arrays)):
        list_of_arrays = [np.concatenate([list_of_arrays[j], arr]) for j, arr in enumerate(lists_of_arrays[i])]

    return list_of_arrays


def compute_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> NDArray[float]:
    ba = a - b
    bc = c - b

    cosine_angles = np.sum(ba * bc, axis=-1) / (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1))
    angles = np.arccos(cosine_angles)

    return np.concatenate([np.degrees(angles), [np.degrees(angles)[-1]]]) if len(a.shape) > 1 else np.degrees(angles)


def transpose_list_of_arrays(list_of_arrays: List[NDArray]) -> List[NDArray]:
    return [np.vstack([arr[i] for arr in list_of_arrays]) for i in range(list_of_arrays[0].shape[0])]


def get_perpendicular_vector(a: NDArray) -> NDArray:
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]


    c = np.empty_like(a)
    c[0] = a[1]
    c[1] = -a[0]

    return np.array([b, c])
