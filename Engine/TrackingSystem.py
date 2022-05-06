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
from Engine.GlobalClock import Agent, TimeTriggeredAgent


class TrackingUnit(Agent):
    """Tracking unit should retrieve position/velocity based on GPS data and upload the data at specified intervals.
    The data should then be received by a corresponding ground system. The system features a list to store data that
    is in transit, as well as RNG-based statistical mtds for determining if/when the data is received by the ground
    system. For main update function, additional pos/vel error agents may be specified to model systems with tracking
    units decoupled from onboard navigation units.
    """
    def __init__(self, AC_ident, update_rate, start_time,
                 latency, latency_distribution=None,
                 availability=1,
                 pos_quant=None,
                 hor_pos_quant=None,
                 vert_pos_quant=None,
                 vel_quant=None,
                 phase_delay=None,
                 rel_clk_sync_err=None
                 ):
        if isinstance(phase_delay, type(None)):
            phase_delay = np.random.uniform(0, 1/update_rate)
        super().__init__(update_rate, start_time, phase_delay)
        self.ident = AC_ident
        self.transit_est_pos_data = []
        self.transit_est_vel_data = []
        self.transmission_time = []
        self.received_time = []
        self.latency = latency
        self.latency_distribution = latency_distribution
        self.availability = availability
        self.pos_quantization = pos_quant  # Number of D.P. each value will be rounded to
        if not isinstance(hor_pos_quant, type(None)):
            self.hor_pos_quant = int(hor_pos_quant)
        else:
            self.hor_pos_quant = hor_pos_quant
        if not isinstance(vert_pos_quant, type(None)):
            self.vert_pos_quant = int(vert_pos_quant)
        else:
            self.vert_pos_quant = vert_pos_quant
        self.vel_quantization = vel_quant  # Number of D.P. each value will be rounded to
        if not isinstance(self.latency_distribution, type(None)):
            latency_func = latency_distribution.get_latency(state=None)
            self.get_latency = latency_func
        self.rel_clk_sync_err = rel_clk_sync_err  # Fixed offset (i.e. no clock drift) clk sync error for timestamp

    def get_latency(self, state=None):
        return self.latency

    def update_and_queue(self, time, state, pos_error_agent=None, vel_error_agent=None):
        """Queues trajectory info for transmission, and generates expected latency OTF. """
        if super().check_time(time):
            if self.availability > np.random.uniform(0, 1):
                if isinstance(pos_error_agent, type(None)):
                    est_pos = state.gt_pos + state.gt_pos_err
                else:
                    est_pos = state.gt_pos + pos_error_agent.update_error(time)

                if isinstance(vel_error_agent, type(None)):
                    est_vel = state.gt_vel + state.gt_vel_err
                else:
                    est_vel = state.gt_vel + vel_error_agent.update_error(time)

                transmission_time = self.next_update_time - self.interval

                if self.hor_pos_quant:
                    est_pos[0], est_pos[1] = round(est_pos[0], self.hor_pos_quant), \
                                             round(est_pos[1], self.hor_pos_quant)
                if self.vert_pos_quant:
                    est_pos[2] = round(est_pos[2], self.vert_pos_quant)

                if self.pos_quantization:
                    est_pos = est_pos.round(self.pos_quantization)

                if self.vel_quantization:
                    est_vel = est_vel.round(self.vel_quantization)

                self.transit_est_pos_data.append(est_pos)
                self.transit_est_vel_data.append(est_vel)
                self.received_time.append(transmission_time + self.get_latency(state))
                # ^^ transmission_time here is "accurate", i.e. synchronized to "global clock"
                # Now, we model effect of de-synchronized clocks between tracking unit
                # and ground station...
                if self.rel_clk_sync_err:
                    self.transmission_time.append(transmission_time + self.rel_clk_sync_err)
                else:
                    self.transmission_time.append(transmission_time)

    def get_next_rec_time(self):
        if len(self.received_time) > 0:
            return self.received_time[0]

    def pop_from_queue(self):
        if len(self.transmission_time) > 0:
            return (self.transmission_time.pop(0), self.received_time.pop(0),
                    self.transit_est_pos_data.pop(0), self.transit_est_vel_data.pop(0))


class SingleTrajectory(object):
    """Stores info for a single A/C trajectory. Main methods are to help retrieve/extrapolate data
    from the stored trajectory. """

    def __init__(self, ident):
        self.ident = ident
        self.trajectory = []

    def update_trajectory(self, transmit_time, received_time, est_pos, est_vel):
        self.trajectory.append(np.array([transmit_time, received_time, *est_pos, *est_vel]))

    def last_known_pos(self):
        if len(self.trajectory) > 0:
            return self.trajectory[-1][2:5]

    def last_known_vel(self):
        if len(self.trajectory) > 0:
            return self.trajectory[-1][5:8]

    def last_transmitted_time(self):
        if len(self.trajectory) > 0:
            return self.trajectory[-1][0]

    def last_received_time(self):
        if len(self.trajectory) > 0:
            return self.trajectory[-1][1]

    def extrapolate_pos(self, extrapolated_time):
        """Linearly extrapolate pos based on last reported A/C position and velocity."""
        if len(self.trajectory) > 0:
            return (extrapolated_time - self.trajectory[-1][0]) * self.trajectory[-1][5:8] + self.trajectory[-1][2:5]

    def extrapolate_pos2(self, extrapolated_time):
        if len(self.trajectory) > 0:
            if self.trajectory[-1][0] > extrapolated_time:
                return self.last_known_pos()
            else:
                return (extrapolated_time - self.trajectory[-1][0]) * self.trajectory[-1][5:8] + self.trajectory[-1][
                                                                                                 2:5]


class GroundStation(TimeTriggeredAgent):
    def __init__(self, update_rate, start_time, phase_delay=0):
        """Class that handles updating of A/C trajectories.
        Note: keys for self.tracked_objects and self.trajectories MUST be the same.
        They should both refer to AC_ident."""
        super().__init__(update_rate, start_time, phase_delay)
        self.tracked_objects = {}
        self.trajectories = {}

    def add_airborne_tracker(self, AC_ident, tracking_unit):
        """Adds new tracking unit to synchronise info with ground tracking station."""
        self.tracked_objects[AC_ident] = tracking_unit
        self.trajectories[AC_ident] = SingleTrajectory(AC_ident)

    def remove_airborne_tracker(self, AC_ident):
        """Note: trajectory is NOT deleted. """
        del self.tracked_objects[AC_ident]

    def track_and_update_trajectories(self, actual_time):
        """Note: rec time generated by the tracking unit is overwritten by actual time of the
        ground system when data is appended to trajectory."""
        for AC_ident, tracking_unit in self.tracked_objects.items():
            if len(tracking_unit.transmission_time) > 0:
                if super().check_time_and_trigger(actual_time, tracking_unit.received_time[0]):
                    trans_time, rec_time1, est_pos, est_vel = tracking_unit.pop_from_queue()
                    rec_time = actual_time
                    self.trajectories[AC_ident].update_trajectory(trans_time, rec_time, est_pos, est_vel)
