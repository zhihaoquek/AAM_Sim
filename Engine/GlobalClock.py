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

    def check_time2(self, time):
        """This does NOT handle updating of next update time. Need to manually change the update time elsewhere
        within the class hierarchy. Useful when there multiple time checks required e.g. tracking system checking
        time for multiple A/C tracking units and a single update interval pass is required. To be used in conjunction
        with self.update_next_update_time()."""
        self.time = time
        return time >= self.next_update_time

    def update_next_update_time(self):
        """To be used with check_time2 to ensure proper update of self interval checking time. """
        self.next_update_time += self.interval


class TimeTriggeredAgent(Agent):
    """An agent that can be triggered at certain given timings.
    Useful when update intervals are not fixed.
    Note: this agent still has an internal update rate, which is used to check
    for when the trigger time is reached. This internal update rate may be set
    to the overall physics/world update rate if desired. """

    def __init__(self, update_rate, start_time, phase_delay=0):
        super().__init__(update_rate, start_time, phase_delay)

    def trigger(self, t):
        my_print('trigger now! time now is: {t1}, trigger time is: {t2}'.format(t1=round(self.time, 4), t2=t))

    def append_to_trigger_timings(self, time):
        self.trigger_time_list.append(time)

    def check_time_and_trigger(self, actual_time, trigger_time):
        if super().check_time2(actual_time):
            if 0 <= (actual_time - trigger_time) / self.interval < 1:
                return True
