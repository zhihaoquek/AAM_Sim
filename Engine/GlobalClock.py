# -*- coding: utf-8 -*-
"""
@Time    : 02/11/2022 9:41 AM
@Author  : Zhi Hao
@FileName: GlobalClock.py
@Description: Controls an overall clock for the simulator. Implements a default time-based agent class that has its own
update rate and start time, and accepts timing control from the global clock.
@Package dependency:
"""
import numpy as np
import CrossPlatformDev
import platform


class GlobalClock(object):
    """Overall class to control sim time."""
    def __init__(self, update_rate, stop, start=0):
        self.time = start
        self.stop = stop
        self.dt = 1/update_rate

    def update(self):
        self.time += self.dt


class Agent(object):
    """Generic agent template. Accepts timing info from global clock. Agent performs tasks at defined intervals."""
    def __init__(self, update_rate, start_time):
        self.update_rate = update_rate
        self.start_time = start_time
        self.interval = 1/self.update_rate
        self.next_update_time = start_time

    def set_next_update_time(self):
        self.next_update_time += self.interval

    def check_time(self, time):
        if time >= self.next_update_time:
            self.next_update_time += self.interval
            return True

