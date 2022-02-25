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
from CrossPlatformDev import my_print
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
    def __init__(self, update_rate, start_time, phase_delay=0):
        self.update_rate = update_rate
        self.start_time = start_time
        self.interval = 1/self.update_rate
        self.next_update_time = start_time + phase_delay
        self.time = start_time

    def set_next_update_time(self):
        self.next_update_time += self.interval

    def check_time(self, time):
        self.time = time
        if time >= self.next_update_time:
            self.next_update_time += self.interval
            return True


class TimeTriggeredAgent(Agent):
    """An agent that can be triggered at certain given timings.
    Useful when update intervals are not fixed.
    Note: this agent still has an internal update rate, which is used to check
    for when the trigger time is reached. This internal update rate may be set
    to the overall physics/world update rate if desired. """

    def __init__(self, update_rate, start_time, phase_delay=0, trigger_time_list=None):
        super().__init__(update_rate, start_time, phase_delay)
        if isinstance(trigger_time_list, type(None)):
            self.trigger_time_list = []
        else:
            self.trigger_time_list = trigger_time_list

    def trigger(self, t):
        my_print('trigger now! time now is: {t1}, trigger time is: {t2}'.format(t1=round(self.time, 4), t2=t))

    def append_to_trigger_timings(self, time):
        self.trigger_time_list.append(time)

    def trigger_time(self, time):
        if super().check_time(time):
            if len(self.trigger_time_list) > 0:
                self.time = time
                if 0 <= (time - self.trigger_time_list[0]) / self.interval < 1:
                    self.trigger(self.trigger_time_list[0])
                    self.trigger_time_list.pop(0)