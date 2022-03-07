# -*- coding: utf-8 -*-
"""
@Time    : 03/7/2022 9:41 AM
@Author  : Zhi Hao
@FileName: TrackingSystem.py
@Description: File containing system to manage I.O. between A/C and ground-based tracking system.
Main features:
Latency, Update Rate, Availability (complement of error rate/transmission failure), Quantization (resolution), etc.
@Package dependency:
"""
import numpy as np
from CrossPlatformDev import my_print
from Engine.GlobalClock import Agent


class TrackingUnit(Agent):
    """Tracking unit should retrieve position/velocity based on GPS data and upload the data at specified intervals.
    The data should then be received by a corresponding ground system. The system features a list to store data that
    is in transit, as well as RNG-based statistical mtds for determining if/when the data is received by the ground
    system. For main update function, additional pos/vel error agents may be specified to model systems with tracking
    units decoupled from onboard navigation units.
    """
    def __init__(self, update_rate, start_time, latency, latency_distribution=None, phase_delay=None):
        if isinstance(phase_delay, type(None)):
            phase_delay = np.random.uniform(0, self.interval)
        super().__init__(update_rate, start_time, phase_delay)
        self.transit_est_pos_data = []
        self.transit_est_vel_data = []
        self.transmission_time = []
        self.received_time = []
        self.latency = latency
        self.latency_distribution = latency_distribution
        if isinstance(self.latency_distribution is not type(None)):
            latency_func = latency_distribution.get_latency(state=None)
            self.get_latency = latency_func


    def get_latency(self, state=None):
        return self.latency

    def update(self, time, state, pos_error_agent=None, vel_error_agent=None):
        """Requires update """
        if super().check_time(time):
            if isinstance(pos_error_agent, type(None)):
                est_pos = state.gt_pos + state.gt_pos_err
            else:
                est_pos = state.gt_pos + pos_error_agent.update_error(time)

            if isinstance(vel_error_agent, type(None)):
                est_vel = state.gt_vel + state.gt_vel_err
            else:
                est_pos = state.gt_vel + vel_error_agent.update_error(time)

            transmission_time = self.next_update_time - self.interval

            self.transit_est_pos_data.append(est_pos)
            self.transit_est_pos_data.append(est_pos)
            self.transmission_time.append(transmission_time)
            self.received_time.append(transmission_time + self.get_latency(state))

