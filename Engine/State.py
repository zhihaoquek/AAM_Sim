# -*- coding: utf-8 -*-
"""
@Time    : 02/11/2022 9:41 AM
@Author  : Zhi Hao
@FileName: State.py
@Description: Records current A/C state.
@Package dependency:
"""
import numpy as np
import CrossPlatformDev
from Engine.GlobalClock import Agent
from scipy.spatial.transform import Rotation as R


class State(Agent):
    def __init__(self, update_rate, start_time, pos, vel, pos_err, vel_err, wind_spd, aircraft_type, rpy=(0, 0, 0)):
        super().__init__(update_rate, start_time)
        self.gt_pos = np.array([*pos])
        self.gt_vel = np.array([*vel])
        self.air_vel = np.array([*vel]) - np.array([*wind_spd])
        self.accel = np.zeros(3)
        self.gt_pos_err = np.array([*pos_err])
        self.gt_vel_err = np.array([*vel_err])
        self.accel_err = np.zeros(3)
        self.rpy = np.array(rpy)
        self.rpy_rate = np.zeros(3)
        self.rpy_accel = np.zeros(3)
        self.aircraft_type = aircraft_type

    def return_dict(self):
        return self.__dict__

    def gt_pos_est(self, time, pos_nav_agent):
        return self.gt_pos + pos_nav_agent.update_error(time)

    def gt_vel_est(self, time, vel_nav_agent):
        return self.gt_vel + vel_nav_agent.update_error(time)

    def update(self, time, drag_model, controller, pos_nav_agent, vel_nav_agent, windfield):
        if super().check_time(time):

            # Update wind and drag
            new_wind_spd = windfield.get_windspd(time, self.gt_pos)
            new_air_vel = self.air_vel - new_wind_spd
            drag = drag_model.get_drag(new_air_vel, self.rpy)

            # Controller calculates thrust, rpy
            target_force, target_rpy = controller.compute(self.gt_pos_est(time, pos_nav_agent),
                                                          self.gt_vel_est(time, vel_nav_agent))

            # Actual thrust/rpy achieved
            cur_rotation = R.from_euler('xyz', self.rpy).as_matrix()
            thrust = np.dot(cur_rotation, target_force)
            achieved_thrust = np.dot(np.array([0, 0, thrust[2]]), cur_rotation.T)
            achieved_accel = (achieved_thrust + drag)/self.aircraft_type.mass \
                             - np.array([0, 0, self.aircraft_type.mass * 9.81])

            target_rpy_accel = controller.pid_attitude(self.rpy, target_rpy)
            new_rpy_rate = self.rpy_rate + target_rpy_accel * controller.interval
            achieved_rpy = (self.rpy_rate + new_rpy_rate) / 2 * controller.interval + self.rpy

            # Update dynamics (physics model)
            new_vel = self.gt_vel + achieved_accel * controller.interval



