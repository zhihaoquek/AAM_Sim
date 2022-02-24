# -*- coding: utf-8 -*-
"""
@Time    : 02/11/2022 9:41 AM
@Author  : Zhi Hao
@FileName: MultiRotorController.py
@Description: Double-loop controller for generic multirotor.
@Package dependency:
"""
import numpy as np
from CrossPlatformDev import my_print
from Engine.GlobalClock import Agent


class MultiRotorController(Agent):
    def __init__(self, update_rate, start_time, flightplan, aircraft_type,
                 wpt_tol_hor, wpt_tol_vert, wpt_tol):
        super().__init__(update_rate, start_time)
        self.net_force = None
        self.controller_error = None
        self.target_rpy = None
        self.target_force = None
        self.aircraft_type = aircraft_type
        self.p_att = np.array([9, 9, 9])  # x, y, z axis
        self.i_att = np.array([0.01, 0.01, 0.01])
        # self.i_att = np.array([0, 0, 0])
        self.d_att = np.array([7, 7, 7])
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)

        self.p_hover = np.array([0.03, 0.03, 0.5])  # x, y, z axis
        self.i_hover = np.array([0, 0, 0])
        self.d_hover = np.array([0.6, 0.6, 2])
        self.hover_pseudo_calculator = PseudoForceCalculator(self.p_hover,
                                                             self.i_hover,
                                                             self.d_hover,
                                                             self.interval)
        self.pid_P2P_tangential = np.array([1, 0, 0.8])
        self.P2P_tangential_pseudo_calculator = PseudoForceCalculator(self.pid_P2P_tangential[0],
                                                                      self.pid_P2P_tangential[1],
                                                                      self.pid_P2P_tangential[2],
                                                                      self.interval, dim=1)
        self.pid_P2P_cross_trk = np.array([0.03, 0, 0.6])
        self.P2P_cross_trk_pseudo_calculator = PseudoForceCalculator(self.pid_P2P_cross_trk[0],
                                                                     self.pid_P2P_cross_trk[1],
                                                                     self.pid_P2P_cross_trk[2],
                                                                     self.interval, dim=1)
        self.pid_P2P_vertical = np.array([0.5, 0, 2])
        self.P2P_vertical_pseudo_calculator = PseudoForceCalculator(self.pid_P2P_vertical[0],
                                                                    self.pid_P2P_vertical[1],
                                                                    self.pid_P2P_vertical[2],
                                                                    self.interval, dim=1)

        self.mode = flightplan.current_leg.get_mode()
        self.flight_plan = flightplan
        self.wpt_tol = wpt_tol
        self.wpt_tol_hor = wpt_tol_hor  # Not used at the moment
        self.wpt_tol_vert = wpt_tol_vert  # Not used at the moment
        self.current_target_speed = None

    def compute(self, est_pos, est_vel, est_accel,  est_wind_spd, time, drag_model, rpy):
        """Computes target force and RPY, depending on controller mode. """

        if super().check_time(time):
            ## my_print('Controller updating target values, time is: ', round(time, 5))

            if self.mode == 'Direct_P2P':  # NOT WORKING YET!!!
                # Check if A/C is is close enough to target position. If yes, change to next leg.
                # Note: More sophisticated flight mode control can be implemented. E.g. if parameter > 1:
                # (i.e. "overshot" target wpt already), then can dynamically insert a new
                # "hover" flight leg to stabilize.
                # Details for such an implementation need to be worked out. E.g. how to exit stabilization mode.
                if np.linalg.norm(est_pos - self.flight_plan.current_leg.get_target_pos()) < self.wpt_tol:
                    if self.flight_plan.change_flight_leg(time) == 'TERMINATE FLIGHT':
                        return self.target_force, self.target_rpy, self.controller_error, self.net_force
                    my_print('Controller mode changed from ' + self.mode + ' to ' + self.flight_plan.current_leg.mode)
                    self.mode = self.flight_plan.current_leg.get_mode()
                    my_print('Current time is: %4d' % (time))
                    return self.compute(est_pos, est_vel, est_accel,  est_wind_spd, time, drag_model, rpy)

                # Convert into transformed FS-like coordinates

                # Tangential (along-trk) error
                tangential_unit_vector = (1 / np.linalg.norm(self.flight_plan.current_leg.hdg)) \
                                         * self.flight_plan.current_leg.hdg
                tangential_vel = np.dot(est_vel, tangential_unit_vector)
                # my_print('tangential_vel: ', tangential_vel)
                tangential_param = self.flight_plan.current_leg.lambda_calculator(est_pos)
                position_error = est_pos - self.flight_plan.current_leg.line_gen(tangential_param)

                if not self.current_target_speed:
                    self.current_target_speed = min(self.flight_plan.current_leg.tgt_speed,
                                                    self.aircraft_type.max_cruise_speed)
                tangential_err = tangential_vel - self.current_target_speed
                # my_print('tangential_err: ', tangential_err)

                # Calculate tangential (along-trk) accel (i.e. along-trk d-term)
                tangential_accel = np.dot(est_accel, tangential_unit_vector)
                # my_print('tangential_accel: ', tangential_accel)

                # Lateral (cross-track) position error:
                horizontal_error = position_error.copy()
                horizontal_error[2] = 0
                cross_trk_error = np.dot(horizontal_error, horizontal_error)
                # my_print('cross_trk_error: ', cross_trk_error)
                # Calculate the horizontal cross-trk velocity (i.e. cross-trk d-term)
                cross_trk_unit_vector = -np.cross(tangential_unit_vector, np.array([0, 0, 1]))
                cross_trk_vel = np.dot(cross_trk_unit_vector, est_vel)
                # my_print('cross_trk_vel: ', cross_trk_vel)

                # Vertical position error:
                vertical_error = position_error[2]
                # my_print('vertical_error:', vertical_error)

                climb_rate = est_vel[2]
                # my_print('climb_rate:', climb_rate)

                # Calculate pseudo-force in FS-like coordinates
                net_force = np.array([0,
                                      0,
                                      self.P2P_vertical_pseudo_calculator.calculate(vertical_error,
                                                                                    climb_rate)])
                # my_print('net_force (stage 1): ', net_force)

                net_force += self.P2P_cross_trk_pseudo_calculator.calculate(cross_trk_error,
                                                                            cross_trk_vel) * cross_trk_unit_vector

                # my_print('net_force (stage 2): ', net_force)

                net_force += self.P2P_tangential_pseudo_calculator.calculate(tangential_err,
                                                                             tangential_accel) * tangential_unit_vector

                # my_print('net_force (stage 3): ', net_force)

                net_force = net_force*self.aircraft_type.mass

                error = np.array([tangential_err, cross_trk_error, vertical_error])

                # Now calculate thrust/force required by factoring drag and gravity
                # Assumes controller has some way of estimating or correcting for drag/gravity
                target_force = net_force - drag_model.get_drag(est_vel + est_wind_spd, rpy) + \
                               self.aircraft_type.mass * np.array([0, 0, 9.81])

            elif self.mode == 'Hover':
                # Check if hover duration is up. If yes, change to next leg.
                if time >= self.flight_plan.current_leg.get_eta():
                    if self.flight_plan.change_flight_leg(time) == 'TERMINATE FLIGHT':
                        return self.target_force, self.target_rpy, self.controller_error, self.net_force
                    my_print('Controller mode changed from ' + self.mode + ' to ' + self.flight_plan.current_leg.mode)
                    self.mode = self.flight_plan.current_leg.get_mode()
                    # return self.compute(est_pos, est_vel, est_accel, est_wind_spd, time, drag_model, rpy)

                # Switch controller tangential target speed back to None. Need this to make sure P2P mode works.
                if self.current_target_speed:
                    self.current_target_speed = None

                error = est_pos - self.flight_plan.current_leg.get_target_pos()
                # # my_print('Controller positional error for Hover is: ', error.copy().round(decimals=4))

                # Net force needed to achieve desired dynamics
                # Note: for PID position error, velocity IS essentially the "D" error term
                net_force = self.aircraft_type.mass * self.hover_pseudo_calculator.calculate(error, est_vel)
                # # my_print('Net force is: ', net_force)

                # Now calculate thrust/force required by factoring drag and gravity
                # Assumes controller has some way of estimating or correcting for drag/gravity
                target_force = net_force - drag_model.get_drag(est_vel + est_wind_spd, rpy) + \
                               self.aircraft_type.mass * np.array([0, 0, 9.81])

            # Force the commanded "z" component of target_force to be greater than (-1g * mass)
            target_force[2] = np.clip(target_force[2], 9.81 * self.aircraft_type.mass * 0.6,
                                      1.8 * 9.81 * self.aircraft_type.mass)

            # Force x and y components of target_force to be clipped based on max RP...
            # Note this is not very satisfactory
            target_force[0] = np.clip(target_force[0],
                                      -self.aircraft_type.mass * 9.81 / np.tan(self.aircraft_type.max_roll_pitch),
                                      self.aircraft_type.mass * 9.81 / np.tan(self.aircraft_type.max_roll_pitch))

            target_force[1] = np.clip(target_force[1],
                                      -self.aircraft_type.mass * 9.81 / np.tan(self.aircraft_type.max_roll_pitch),
                                      self.aircraft_type.mass * 9.81 / np.tan(self.aircraft_type.max_roll_pitch))

            # Calculate desired rotation. From original UAMTrafficSimulator.
            target_rpy = np.zeros(3)
            sign_z = np.sign(target_force[2])
            if not sign_z.all:
                sign_z = 1

            # # my_print(target_force)
            # # my_print(sign_z)

            # This method of clipping is not entirely satisfactory. There should be max angle with horizon i.e.
            # tilt that should be allowed. Separate clipping of x/y axis means max tilt for off-x/y axis rotation
            # is actually higher.
            target_rpy[0] = np.arcsin(-sign_z * target_force[1] / np.linalg.norm(target_force))
            target_rpy[1] = np.arctan2(sign_z * target_force[0], sign_z * target_force[2])
            target_rpy[2] = 0.
            target_rpy[0] = np.clip(target_rpy[0], -self.aircraft_type.max_roll_pitch, self.aircraft_type.max_roll_pitch)
            target_rpy[1] = np.clip(target_rpy[1], -self.aircraft_type.max_roll_pitch, self.aircraft_type.max_roll_pitch)

            # Note: controller error and net force are used mainly for debugging. Can be deleted in the future. Rmb to
            # edit the "State.py" file as well
            self.target_force = target_force
            self.target_rpy = target_rpy
            self.controller_error = error
            self.net_force = net_force

        return self.target_force, self.target_rpy, self.controller_error, self.net_force

    def pid_attitude(self, cur_rpy, target_rpy):
        """Simple PID attitude control (with yaw fixed to 0). From UAMTrafficSimulator.

        Parameters
        ----------
        control_timestep : float
            The time step at which control is computed.
        cur_rpy : ndarray
            (3,1)-shaped array of floats containing the current orientation in roll, pitch and yaw.
        target_rpy : ndarray
            (3,1)-shaped array of floats containing the computed the target orientation in roll, pitch, and yaw.

        Returns
        -------
        ndarray
            (3,1)-shaped array of floats containing target angular acceleration.

        """
        # cur_rpy = p.getEulerFromQuaternion(cur_quat)
        rpy_e = target_rpy - np.array(cur_rpy).reshape(3, )
        if rpy_e[2] > np.pi:
            rpy_e[2] = rpy_e[2] - 2 * np.pi
        if rpy_e[2] < -np.pi:
            rpy_e[2] = rpy_e[2] + 2 * np.pi
        d_rpy_e = (rpy_e - self.last_rpy_e) / self.interval
        self.last_rpy_e = rpy_e
        self.integral_rpy_e = self.integral_rpy_e + rpy_e * self.interval
        #### PID target torques ####################################
        target_angular_accel = np.multiply(self.p_att, rpy_e) \
                               + np.multiply(self.i_att, self.integral_rpy_e) \
                               + np.multiply(self.d_att, d_rpy_e)

        return target_angular_accel


class PseudoForceCalculator(object):
    """Calculates a pseudo-force/acceleration in transformed coordinates corresponding,
    similar to Frenet-Serret frame. This "force" will then be transformed back into the body-frame. """

    def __init__(self, p, i, d, interval, dim=3):
        self.p = p
        self.i = i
        self.d = d
        self.interval = interval
        if dim == 1:
            self.err = 0
            self.d_err = 0
            self.i_err = 0
        else:
            self.err = np.zeros(dim)
            self.d_err = np.zeros(dim)
            self.i_err = np.zeros(dim)

    def calculate(self, err, d_err=None):
        self.i_err += err
        self.d_err = (err - self.err) / self.interval
        self.err = err
        if d_err is None:
            return -(self.p * self.err + self.i * self.i_err + self.d * self.d_err)
        else:
            return -(self.p * self.err + self.i * self.i_err + self.d * d_err)

    def reset_i_err(self):
        self.i_err = 0
