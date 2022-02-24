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
    if platform.system() == 'Windows':
        print(*args)


