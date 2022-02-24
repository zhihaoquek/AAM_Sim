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
import pandas as pd
from Engine.GlobalClock import Agent
from scipy.spatial.transform import Rotation as R


class State(Agent):
    def __init__(self, update_rate, start_time, pos, vel, pos_err, vel_err, wind_spd, aircraft_type, rpy=(0, 0, 0)):
        super().__init__(update_rate, start_time)
        self.gt_pos = np.array([*pos])
        self.gt_vel = np.array([*vel])
        self.wind_spd = np.array([*wind_spd])
        self.air_vel = np.array([*vel]) - np.array([*wind_spd])
        self.accel = np.zeros(3)
        self.gt_pos_err = np.array([*pos_err])
        self.gt_vel_err = np.array([*vel_err])
        self.accel_err = np.zeros(3)
        self.rpy = np.array(rpy)
        self.rpy_rate = np.zeros(3)
        self.rpy_accel = np.zeros(3)
        self.aircraft_type = aircraft_type
        self.thrust = np.zeros(3)
        self.time = start_time
        self.commanded_net_force = np.zeros(3)
        self.controller_pos_err = np.zeros(3)
        self.trajectory = []



    def return_dict(self):
        return self.__dict__

    def return_df(self):
        return pd.DataFrame({'gt_pos':[self.gt_pos], 'gt_vel':[self.gt_vel], 'wind_spd':[self.wind_spd],
                             'air_vel':[self.air_vel], 'accel':[self.accel],
                             'gt_pos_err':[self.gt_pos_err], 'gt_vel_err':[self.gt_vel_err],
                             'accel_err':[self.accel_err],
                             'rpy':[self.rpy], 'rpy_rate':[self.rpy_rate], 'rpy_accel':[self.rpy_accel],
                             'commanded_net_force':[self.commanded_net_force],
                             'controller_pos_err':[self.controller_pos_err],
                             'thrust':[self.thrust],
                             'time':[self.time]
                             })

    def gt_pos_est(self, time, pos_nav_agent):
        self.gt_pos_err = pos_nav_agent.update_error(time)
        return self.gt_pos + self.gt_pos_err

    def gt_vel_est(self, time, vel_nav_agent):
        self.gt_vel_err = vel_nav_agent.update_error(time)
        return self.gt_vel + self.gt_vel_err

    def gt_accel_est(self, time, accel_nav_agent):
        self.accel_err = accel_nav_agent.update_error(time)
        return self.accel + self.accel_err

    def air_vel_est(self, time, air_vel_sensor):
        return self.air_vel

    def update(self, time, drag_model, controller, pos_nav_agent, vel_nav_agent, air_speed_sensor, windfield):
        if super().check_time(time):

            # Update wind and drag
            new_wind_spd = windfield.get_windspd(time, self.gt_pos)
            new_air_vel = self.gt_vel - new_wind_spd
            drag = drag_model.get_drag(new_air_vel, self.rpy)

            # Controller calculates thrust, rpy
            target_force, target_rpy, controller_error, commanded_net_force = controller.compute(self.gt_pos_est(time, pos_nav_agent),
                                                                                                 self.gt_vel_est(time, vel_nav_agent),
                                                                                                 new_wind_spd, time, drag_model, self.rpy)

            # Actual thrust/rpy achieved
            cur_rotation = R.from_euler('xyz', self.rpy).as_matrix()
            thrust = np.dot(cur_rotation, target_force)
            achieved_thrust = np.dot(np.array([0, 0, thrust[2]]), cur_rotation.T)
            achieved_accel = (achieved_thrust + drag)/self.aircraft_type.mass \
                             - np.array([0, 0, self.aircraft_type.mass * 9.81])

            target_rpy_accel = controller.pid_attitude(self.rpy, target_rpy)
            new_rpy_rate = self.rpy_rate + target_rpy_accel * self.interval
            achieved_rpy = (self.rpy_rate + new_rpy_rate) / 2 * self.interval + self.rpy

            # Update dynamics (physics model)
            new_vel = self.gt_vel + achieved_accel * self.interval
            new_gt_pos = self.gt_pos + new_vel * self.interval

            # Update all state values (note: error values are automatically updated when estimate methods are called)
            # Note: to test if removal of copy still provides correct values. If ok, might be better to remove for
            # better performance.
            self.gt_pos = new_gt_pos.copy()
            self.gt_vel = new_vel.copy()
            self.wind_spd = new_wind_spd.copy()
            self.air_vel = new_air_vel.copy()
            self.thrust = achieved_thrust.copy()
            self.accel = achieved_accel.copy()
            self.rpy_accel = target_rpy_accel.copy()
            self.rpy_rate = new_rpy_rate.copy()
            self.rpy = achieved_rpy.copy()
            self.time = time
            self.commanded_net_force = commanded_net_force.copy()
            self.controller_pos_err = controller_error.copy()


            self.trajectory.append(self.return_df().copy())


    def get_trajectory(self):
        if len(self.trajectory) != 0:
            return pd.concat(self.trajectory)
        else:
            return None


