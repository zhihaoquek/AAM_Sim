"""
@Time    : 02/11/2022 9:41 AM
@Author  : Zhi Hao
@FileName: CrossPlatformDev.py
@Description: Controls an overall clock for the simulator. Implements a default time-based agent class that has its own
update rate and start time, and accepts timing control from the global clock.
@Package dependency:
"""
import platform
from scipy import interpolate
import pandas as pd
import numpy as np

def my_print(*args):
    """Just to make things a little prettier :)
    For debugging use. Prints output when debugging on a Windows machine.
    For running on Linux systems like AWS servers, it will NOT print. This is to avoid
    issues with tqdm package in Jupyter notebooks. """
    if platform.system() == 'Windows':
        print(*args)


def join_str(*args):
    if platform.system() == 'Windows':
        if len(args) > 1:
            return args[0] + '\\' + join_str(*args[1:])
        else:
            return args[0]
    elif platform.system() == 'Linux':
        if len(args) > 1:
            return args[0] + '/' + join_str(*args[1:])
        else:
            return args[0]


def keyword_list_filter(l, *keys):
    if len(keys) > 1:
        # print(keys[0])
        l_filtered = list(filter(lambda x: keys[0] in x, l))
        # print('l_filtered', l_filtered, 'keys[1:]', *keys[1:])
        return keyword_list_filter(l_filtered, *keys[1:])
    else:
        # print(keys[0])
        return list(filter(lambda x: keys[0] in x, l))


def keyword_list_exclude(l, *keys):
    if len(keys) > 1:
        l_filtered = list(filter(lambda x: keys[0] not in x, l))
        return keyword_list_exclude(l_filtered, *keys[1:])
    else:
        return list(filter(lambda x: keys[0] not in x, l))


def filter_column_names(df, *keys):
    cols = df.columns
    return keyword_list_filter(cols, *keys)

def param_search(func, search_space, est_pos, est_sigma=None):
    """Estimates spline parameter.
    Args: func, search_space, est_pos, est_sigma=None
    func: the object supplied to interpolate.splev that defines spline shape
    search_space: numpy array containing values of params you wish to check
    est_pos: estimated position that you wish to search around
    est_sigma: initial guess of spline parameter"""
    x, y, z = interpolate.splev(search_space, func)
    est_x, est_y, est_z = est_pos
    x_dist = est_x - x
    y_dist = est_y - y
    z_dist = est_z - z
    mini_df = pd.DataFrame({'x': x, 'y': y, 'z': z, 'dist_to_Spline': np.sqrt(x_dist**2 + y_dist**2 + z_dist**2),
                            'sigma': search_space})
    if not isinstance(est_sigma, type(None)):
        mini_df.sort_values(by=['dist_to_Spline'], inplace=True)
        mini_df = mini_df.head(10)
        # delta_sigma = sigma
        # mini_df = mini_df[mini_df['dist_to_Spline']==mini_df['dist_to_Spline'].min()]
        mini_df['delta_sigma'] = (est_sigma - mini_df['sigma'])**2
        mini_df.sort_values(by=['delta_sigma'], inplace=True)
    else:
        mini_df = mini_df[mini_df['dist_to_Spline'] == mini_df['dist_to_Spline'].min()]
    clst_x = mini_df.iloc[0]['x']
    clst_y = mini_df.iloc[0]['y']
    clst_z = mini_df.iloc[0]['z']
    clst_sigma = mini_df.iloc[0]['sigma']
    return clst_x, clst_y, clst_z, clst_sigma


def auto_scale(func, ds, approx_path_len, coverg_tgt, target_ds_res):
    spl_s_new = np.arange(0, 1 + ds, ds)
    path_dx_dy_dz = interpolate.splev(spl_s_new, func, der=1)
    new_path_len = np.sum(np.sqrt(path_dx_dy_dz[0]**2 + path_dx_dy_dz[1]**2 + path_dx_dy_dz[2]**2)) * ds
    if abs(new_path_len - approx_path_len)/approx_path_len < coverg_tgt:
        target_ds = target_ds_res/new_path_len
        return {'ds_path_len':ds,
                'Path_len':new_path_len,
                'ds_target_res':target_ds}
    else:
        return auto_scale(func, ds/10, new_path_len, coverg_tgt, target_ds_res)