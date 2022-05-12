"""
@Time    : 02/11/2022 9:41 AM
@Author  : Zhi Hao
@FileName: CrossPlatformDev.py
@Description: Controls an overall clock for the simulator. Implements a default time-based agent class that has its own
update rate and start time, and accepts timing control from the global clock.
@Package dependency:
"""
import platform


def my_print(*args):
    """Just to make things a little prettier :)
    For debugging use. Prints output when debugging on a Windows machine.
    For running on Linux systems like AWS servers, it will NOT print. This is to avoid
    issues with tqdm package in Jupyter notebooks. """
    #if platform.system() == 'Windows':
        #print(*args)


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