# -*- coding: utf-8 -*-
"""
@Time    : 02/11/2022 9:41 AM
@Author  : Zhi Hao
@FileName: MultiRotorController.py
@Description: Double-loop controller for generic multirotor.
@Package dependency:
"""
import numpy as np
from CrossPlatformDev import my_print, auto_scale, param_search
from Engine.GlobalClock import Agent
from scipy import interpolate


class MultiRotorController(Agent):
    def __init__(self, update_rate, start_time, flightplan, aircraft_type,
                 wpt_tol_hor, wpt_tol_vert, wpt_tol):
        super().__init__(update_rate, start_time)
        self.net_force = None
        self.controller_error = None
        self.target_rpy = None
        self.target_force = None
        self.aircraft_type = aircraft_type
        self.p_att = np.array([0.1, 0.1, 0.1])  # x, y, z axis
        self.i_att = np.array([0, 0, 0])
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
        # Note: be careful when adjusting tangential I values. If commanded tangential force increases
        # more quickly than commanded vertical force, RPY controller may "lock" at max RPY value.
        self.pid_P2P_tangential = np.array([1, 0.003*120, 1])
        self.P2P_tangential_pseudo_calculator = PseudoForceCalculator(self.pid_P2P_tangential[0],
                                                                      self.pid_P2P_tangential[1],
                                                                      self.pid_P2P_tangential[2],
                                                                      self.interval, dim=1)
        # self.pid_P2P_cross_trk = np.array([0.03, 0, 0.6])  # Previous best
        self.pid_P2P_cross_trk = np.array([2, 0*120, 1.5])  # For testing
        self.P2P_cross_trk_pseudo_calculator = PseudoForceCalculator(self.pid_P2P_cross_trk[0],
                                                                     self.pid_P2P_cross_trk[1],
                                                                     self.pid_P2P_cross_trk[2],
                                                                     self.interval, dim=1)
        self.pid_P2P_vertical = np.array([6, 0.0035*120, 2.5])
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
        self.tangential_unit_vector = None
        self.tangential_unit_vector_planar = None
        self.cross_trk_unit_vector = None
        self.max_horizontal_force = self.aircraft_type.mass * 9.81 * np.tan(self.aircraft_type.max_roll_pitch)

    def compute(self, est_pos, est_vel, est_accel,  est_wind_spd, time, drag_model, rpy):
        """Computes target force and RPY, depending on controller mode. """

        if super().check_time(time):
            # my_print('Controller updating target values, time is: ', round(time, 5))

            if self.mode == 'Direct_P2P':  # NOTE: NOT WORKING FOR VERTICAL ASCENT/DESCENT
                # Check if A/C is is close enough to target position. If yes, change to next leg.
                # Note: More sophisticated flight mode control can be implemented. E.g. if parameter > 1:
                # (i.e. "overshot" target wpt already), then can dynamically insert a new
                # "hover" flight leg to stabilize.
                # Details for such an implementation need to be worked out. E.g. how to exit stabilization mode.

                # Tangential param estimation. Used for WPT change logic.
                # Saves some re-used params to memory for better performance.
                if isinstance(self.tangential_unit_vector, type(None)):
                    self.tangential_unit_vector = (1 / np.linalg.norm(self.flight_plan.current_leg.hdg)) \
                                                  * self.flight_plan.current_leg.hdg
                    hdg = self.flight_plan.current_leg.hdg.copy()
                    hdg[2] = 0
                    self.tangential_unit_vector_planar = (1/np.linalg.norm(hdg)) * hdg
                    # self.cross_trk_unit_vector = -np.cross(self.tangential_unit_vector_planar, np.array([0, 0, 1]))

                # hdg = self.flight_plan.current_leg.hdg.copy()
                # hdg[2] = 0
                # tangential_unit_vector_planar = (1/np.linalg.norm(hdg)) * hdg
                tangential_vel = np.dot(est_vel, self.tangential_unit_vector_planar)
                # my_print('tangential_vel: ', tangential_vel)
                tangential_param = self.flight_plan.current_leg.lambda_calculator(est_pos)

                # if np.linalg.norm(est_pos - self.flight_plan.current_leg.get_target_pos()) < self.wpt_tol:
                # ^^ uses Euclidean distance between est. position and tgt wpt to change flight leg.
                # but ^^ is unstable as A/C may "pass" wpt without being close enough to trigger flight leg change.
                # can use tangential param instead to trigger change, i.e. once aircraft reaches/"overshoots" tgt wpt.
                if tangential_param >= 1:
                    if self.flight_plan.change_flight_leg(time) == 'TERMINATE FLIGHT':
                        return self.target_force, self.target_rpy, self.controller_error, self.net_force
                    # my_print('Controller mode changed from ' + self.mode + ' to ' + self.flight_plan.current_leg.mode)
                    self.mode = self.flight_plan.current_leg.get_mode()
                    self.reset_tangential_target_speed_hdg()
                    # my_print('Current time is: %4d' % (time))
                    return self.compute(est_pos, est_vel, est_accel,  est_wind_spd, time, drag_model, rpy)

                # Convert into transformed FS-like coordinates

                # Tangential (along-trk) error
                position_error = est_pos - self.flight_plan.current_leg.line_gen(tangential_param)

                if not self.current_target_speed:
                    self.current_target_speed = min(self.flight_plan.current_leg.tgt_speed,
                                                    self.aircraft_type.max_cruise_speed)
                    self.current_target_speed = np.sqrt(1 - self.tangential_unit_vector[2]**2) \
                                                * self.current_target_speed
                tangential_err = tangential_vel - self.current_target_speed
                # my_print('tangential_err (velocity): {x}, time: {t}'.format(x=tangential_err, t=time))

                # Calculate tangential (along-trk) accel (i.e. along-trk d-term)
                tangential_accel = np.dot(est_accel, self.tangential_unit_vector_planar)
                # my_print('tangential_accel: ', tangential_accel)

                # Lateral (cross-track) position error:
                horizontal_error = position_error.copy()
                cross_trk_unit_vector = -np.cross(self.tangential_unit_vector_planar, np.array([0, 0, 1]))
                cross_trk_error = np.dot(horizontal_error, cross_trk_unit_vector)
                # my_print('cross_trk_error: ', cross_trk_error)
                # Calculate the horizontal cross-trk velocity (i.e. cross-trk d-term)
                cross_trk_vel = np.dot(cross_trk_unit_vector, est_vel)
                # my_print('cross_trk_vel: ', cross_trk_vel)
                # Vertical position error:
                vertical_error = position_error[2]
                # my_print('vertical_error: {v}, time: {t}'.format(v=vertical_error, t=time))
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
                                                                             tangential_accel) \
                             * self.tangential_unit_vector_planar

                # net_force += self.P2P_tangential_pseudo_calculator.calculate(tangential_err) \
                #              * self.tangential_unit_vector_planar

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
                    self.reset_tangential_target_speed_hdg()  # Reset target speeds/hdgs/unit vectors
                    return self.compute(est_pos, est_vel, est_accel, est_wind_spd, time, drag_model, rpy)

                # Switch controller tangential target speed and hdg back to None. Need this to make sure P2P mode works.
                # self.reset_tangential_target_speed_hdg()

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
                                      2.5 * 9.81 * self.aircraft_type.mass)

            # Force x and y components of target_force to be clipped based on max RP...
            # There is some error with this mtd of calculating...
            # horizontal_force = target_force[0]**2 + target_force[1]**2
            # if horizontal_force > (self.max_horizontal_force * 1)**2:
            #     my_print("Horizontal force reaching max at time %.3f" % time)
            #     horizontal_force_scale = ((self.max_horizontal_force * 1)**2)/horizontal_force
            #     target_force[0] = target_force[0] * horizontal_force_scale
            #     target_force[1] = target_force[1] * horizontal_force_scale

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

            # if abs(target_force[0]) >= self.aircraft_type.mass * 9.81 * np.tan(self.aircraft_type.max_roll_pitch):
            #     my_print('X force component saturated at %.3f sec, tgt pre-clip P: %.3f '%(self.time, target_rpy[1]))

            target_rpy[0] = np.clip(target_rpy[0],
                                    -self.aircraft_type.max_roll_pitch,
                                    self.aircraft_type.max_roll_pitch)
            target_rpy[1] = np.clip(target_rpy[1],
                                    -self.aircraft_type.max_roll_pitch,
                                    self.aircraft_type.max_roll_pitch)

            # Note: controller error and net force are used mainly for debugging. Can be deleted in the future. Rmb to
            # edit the "State.py" file as well
            self.target_force = target_force
            self.target_rpy = target_rpy
            self.controller_error = error
            self.net_force = net_force

        return self.target_force, self.target_rpy, self.controller_error, self.net_force

    def reset_tangential_target_speed_hdg(self):
        """Called whenever there is a change in flight leg. Resets some of values. """
        self.current_target_speed = None
        self.tangential_unit_vector = None
        self.tangential_unit_vector_planar = None
        self.cross_trk_unit_vector = None
        # self.P2P_tangential_pseudo_calculator.reset_i_err()
        self.P2P_cross_trk_pseudo_calculator.reset_i_err()
        # self.P2P_vertical_pseudo_calculator.reset_i_err()
        self.hover_pseudo_calculator.reset_i_err()
        # return None

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
        # ---- PID target torques ----
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
        self.i_err += err * self.interval
        self.d_err = (err - self.err) / self.interval
        self.err = err
        if d_err is None:
            return -(self.p * self.err + self.i * self.i_err + self.d * self.d_err)
        else:
            return -(self.p * self.err + self.i * self.i_err + self.d * d_err)

    def reset_i_err(self):
        self.i_err = 0


class MR_Controller(Agent):
    def __init__(self, update_rate, start_time, flightplan, aircraft_type,
                 wpt_tol_hor, wpt_tol_vert, wpt_tol):
        super().__init__(update_rate, start_time)
        self.p_att = np.array([0.1, 0.1, 0.1])  # x, y, z axis
        self.i_att = np.array([0, 0, 0])
        # self.i_att = np.array([0, 0, 0])
        self.d_att = np.array([7, 7, 7])
        self.last_rpy_e = np.zeros(3)
        self.integral_rpy_e = np.zeros(3)
        self.mission_mode = 'Auto'
        self.net_force = None
        self.controller_error = None
        self.target_rpy = None
        self.target_force = None
        self.aircraft_type = aircraft_type
        self.mode = flightplan.current_leg.get_mode()
        self.flight_plan = flightplan
        self.wpt_tol = wpt_tol
        self.wpt_tol_hor = wpt_tol_hor  # Not used at the moment
        self.wpt_tol_hor_sq = wpt_tol_hor**2
        self.wpt_tol_vert = wpt_tol_vert  # Not used at the moment
        self.max_horizontal_force = self.aircraft_type.mass * 9.81 * np.tan(self.aircraft_type.max_roll_pitch)
        my_print(self.flight_plan.current_leg.mode)
        # if self.flight_plan.current_leg.mode == 'Hover':
        #     self.tgt_hdg = np.zeros(3)  # This is a vector in Cartesian, not a HDG angle
        # else:
        #     # self.tgt_hdg = self.flight_plan.current_leg['hdg']
        self.spl_param = 0
        self.spl_params = []
        self.spl_times = []

        self.vert_pos_PID_hover_cntrl = PID(self.update_rate, 4, 0.1, 50)
        self.hor_hover_PID_cntrl_x = PID(self.update_rate, 1, 0, 10)
        self.hor_hover_PID_cntrl_y = PID(self.update_rate, 1, 0, 10)

        self.hor_rate_pos_PID_cntrl = PID(self.update_rate, 3, 0.7, 5)
        self.hor_cross_trk_pos_PID_cntrl = PID(self.update_rate, 1, 0, 10)
        self.vert_pos_PID_moving_cntrl = PID(self.update_rate, 6, 0.1, 50)

        self.vert_rate_PID_cntrl = PID(self.update_rate, 0.3, 0, 0.5)

        self.hor_spline_cross_trk_PID_cntrl = PID(self.update_rate, 3, 0.1, 10)

        self.stab_s_spd = 2.5


    def vert_pos_PID_hover(self, vert_dev):
        net_vertical_force = self.vert_pos_PID_hover_cntrl.cntrl(vert_dev)
        return net_vertical_force

    def vert_pos_PID_moving(self, vert_dev):
        net_vertical_force = self.vert_pos_PID_moving_cntrl.cntrl(vert_dev)
        return net_vertical_force

    def vert_rate_PID(self, vert_rate_dev):
        net_vertical_force = self.vert_rate_PID_cntrl.cntrl(vert_rate_dev)
        return net_vertical_force

    def hor_cross_trk_pos_PID(self, hor_dev):
        net_hor_cross_trk_force = self.hor_cross_trk_pos_PID_cntrl.cntrl(hor_dev)
        return net_hor_cross_trk_force

    def hor_rate_pos_PID(self, hor_vert_dev):
        net_hor_rate_force = self.hor_rate_pos_PID_cntrl.cntrl(hor_vert_dev)
        return net_hor_rate_force

    def hor_hover_PID(self, hor_dev_x, hor_dev_y):
        net_hor_hover_force_x = self.hor_hover_PID_cntrl_x.cntrl(hor_dev_x)
        net_hor_hover_force_y = self.hor_hover_PID_cntrl_y.cntrl(hor_dev_y)
        return net_hor_hover_force_x, net_hor_hover_force_y

    def spline_cross_trk_PID(self, hor_dev):
        return self.vert_rate_PID_cntrl.cntrl(hor_dev)

    def change_mission_mode(self, mode):
        self.mission_mode = mode

    def set_manual_hdg(self, hdg_vec):
        self.tgt_hdg = hdg_vec

    def compute(self, est_pos, est_vel, est_accel, est_wind_spd, time, drag_model, rpy):
        if super().check_time(time):
            if self.mission_mode == 'Manual':
                # Do something in manual mode...
                a = 10
            elif self.mode == 'Hover':
                # Check if hover duration is up. If yes, change to next leg.
                if time >= self.flight_plan.current_leg.get_eta():
                    if self.flight_plan.change_flight_leg(time) == 'TERMINATE FLIGHT':
                        return self.compute(est_pos, est_vel, est_accel, est_wind_spd, time, drag_model, rpy)
                    my_print('Controller mode changed from ' + self.mode + ' to ' + self.flight_plan.current_leg.mode
                             + ' at time %.3f' % time)
                    self.mode = self.flight_plan.current_leg.get_mode()
                    self.vert_pos_PID_hover_cntrl.reset()
                    self.hor_hover_PID_cntrl_x.reset()
                    self.hor_hover_PID_cntrl_y.reset()
                dev = self.flight_plan.current_leg.get_target_pos() - est_pos
                net_hor_hover_force_x, net_hor_hover_force_y = self.hor_hover_PID(dev[0], dev[1])
                net_vertical_force = self.vert_pos_PID_hover(dev[2])
                net_force = np.array([net_hor_hover_force_x, net_hor_hover_force_y, net_vertical_force])

            elif self.mode == 'Direct_P2P':
                # Check if AC is near its target waypoint. If yes, change to next leg.
                dist_to_end_wpt = self.flight_plan.current_leg.get_target_pos() - est_pos

                hor_dev_sq = dist_to_end_wpt[0]**2 + dist_to_end_wpt[1]**2
                param = self.flight_plan.current_leg.lambda_calculator(est_pos)
                if (self.wpt_tol_hor_sq >= hor_dev_sq) & (self.wpt_tol_vert > abs(dist_to_end_wpt[2])):
                    if self.flight_plan.change_flight_leg(time) == 'TERMINATE FLIGHT':
                        return self.compute(est_pos, est_vel, est_accel, est_wind_spd, time, drag_model, rpy)
                    my_print('Controller mode changed from ' + self.mode + ' to ' + self.flight_plan.current_leg.mode
                             + ' at time %.3f' % time)
                    self.mode = self.flight_plan.current_leg.get_mode()
                    ###### RST PIDs
                    self.hor_rate_pos_PID_cntrl.reset()
                    self.hor_cross_trk_pos_PID_cntrl.reset()
                    self.vert_pos_PID_moving_cntrl.reset()
                elif (((self.flight_plan.current_leg.tgt_speed**2 - self.stab_s_spd**2)/(2*0.3*9.81) + self.wpt_tol_hor
                       >= np.sqrt(hor_dev_sq)) & (20 > abs(dist_to_end_wpt[2]))) | (param >= 1):
                    self.mode = 'Stabilize_S'
                    my_print('Controller entering Stabilize_S Mode at time %.3f'%time)
                    return self.compute(est_pos, est_vel, est_accel, est_wind_spd, time, drag_model, rpy)

                clst_trk_pos = param * self.flight_plan.current_leg.hdg + self.flight_plan.current_leg.starting_pos
                dev = clst_trk_pos - est_pos
                # dev1 = dev.copy()
                vert_dev = dev[2]
                net_vert_force = self.vert_pos_PID_moving(vert_dev)
                # dev1[2] = 0
                hor_cross_trk_dev = self.flight_plan.current_leg.cross_trk @ dev
                net_hor_cross_trk_force = self.flight_plan.current_leg.cross_trk * \
                                          self.hor_cross_trk_pos_PID(hor_cross_trk_dev)
                along_trk_dev = self.flight_plan.current_leg.tgt_speed - \
                                (self.flight_plan.current_leg.along_trk @ est_vel)
                dev = np.array([along_trk_dev, hor_cross_trk_dev, vert_dev])
                # if time == 0:
                # my_print('Speed Dev is: %.2f'%along_trk_dev)
                net_hor_along_trk_force = self.flight_plan.current_leg.along_trk * \
                                          self.hor_rate_pos_PID(along_trk_dev)

                net_force = np.array([0, 0, net_vert_force]) + net_hor_cross_trk_force + net_hor_along_trk_force


            elif self.mode == 'Stabilize_S':
                # Check if AC is near its target waypoint. If yes, change to next leg.
                dist_to_end_wpt = self.flight_plan.current_leg.get_target_pos() - est_pos

                hor_dev_sq = dist_to_end_wpt[0]**2 + dist_to_end_wpt[1]**2
                param = self.flight_plan.current_leg.lambda_calculator(est_pos)
                if ((25 >= hor_dev_sq) & (5 > abs(dist_to_end_wpt[2]))) | (param >= 1):
                    # if self.flight_plan.change_flight_leg(time) == 'TERMINATE FLIGHT':
                    #     return self.compute(est_pos, est_vel, est_accel, est_wind_spd, time, drag_model, rpy)
                    # my_print('Controller mode changed from ' + self.mode + ' to ' + self.flight_plan.current_leg.mode
                    #          + ' at time %.3f' % time)
                    # self.mode = self.flight_plan.current_leg.get_mode()
                    # ###### RST PIDs
                    # self.hor_rate_pos_PID_cntrl.reset()
                    # self.hor_cross_trk_pos_PID_cntrl.reset()
                    # self.vert_pos_PID_moving_cntrl.reset()
                # elif ((25 >= hor_dev_sq) & (5 > abs(dist_to_end_wpt[2]))) | (param >= 1):
                    self.mode = 'Stabilize'
                    self.hor_rate_pos_PID_cntrl.reset()
                    self.hor_cross_trk_pos_PID_cntrl.reset()
                    self.vert_pos_PID_moving_cntrl.reset()
                    my_print('Controller entering Stabilize Mode at time %.3f' % time)
                    return self.compute(est_pos, est_vel, est_accel, est_wind_spd, time, drag_model, rpy)

                clst_trk_pos = param * self.flight_plan.current_leg.hdg + self.flight_plan.current_leg.starting_pos
                dev = clst_trk_pos - est_pos
                # dev1 = dev.copy()
                vert_dev = dev[2]
                net_vert_force = self.vert_pos_PID_moving(vert_dev)
                # dev1[2] = 0
                hor_cross_trk_dev = self.flight_plan.current_leg.cross_trk @ dev
                net_hor_cross_trk_force = self.flight_plan.current_leg.cross_trk * \
                                          self.hor_cross_trk_pos_PID(hor_cross_trk_dev)
                along_trk_dev = self.stab_s_spd - \
                                (self.flight_plan.current_leg.along_trk @ est_vel)
                # if time == 0:
                # my_print('Speed Dev is: %.2f'%along_trk_dev)
                dev = np.array([along_trk_dev, hor_cross_trk_dev, vert_dev])
                net_hor_along_trk_force = self.flight_plan.current_leg.along_trk * \
                                          self.hor_rate_pos_PID(along_trk_dev)


                net_force = np.array([0, 0, net_vert_force]) + net_hor_cross_trk_force + net_hor_along_trk_force

            elif self.mode == 'Stabilize':
                # Check if AC is near its target waypoint. If yes, change to next leg.
                dist_to_end_wpt = self.flight_plan.current_leg.get_target_pos() - est_pos
                hor_dev_sq = dist_to_end_wpt[0]**2 + dist_to_end_wpt[1]**2
                if (self.wpt_tol_hor_sq >= hor_dev_sq) & (self.wpt_tol_vert > abs(dist_to_end_wpt[2])):
                    if self.flight_plan.change_flight_leg(time) == 'TERMINATE FLIGHT':
                        return self.compute(est_pos, est_vel, est_accel, est_wind_spd, time, drag_model, rpy)
                    self.vert_pos_PID_hover_cntrl.reset()
                    self.hor_hover_PID_cntrl_x.reset()
                    self.hor_hover_PID_cntrl_y.reset()
                    my_print('Controller mode changed from ' + self.mode + ' to ' + self.flight_plan.current_leg.mode
                             + ' at time %.3f'%time)
                    self.mode = self.flight_plan.current_leg.get_mode()
                dev = self.flight_plan.current_leg.get_target_pos() - est_pos
                net_hor_hover_force_x, net_hor_hover_force_y = self.hor_hover_PID(dev[0], dev[1])
                net_vertical_force = self.vert_pos_PID_hover(dev[2])
                net_force = np.array([net_hor_hover_force_x, net_hor_hover_force_y, net_vertical_force])

            elif self.mode == 'Climb':
                # Check if AC is near its target waypoint. If yes, change to next leg.
                dev = self.flight_plan.current_leg.get_target_pos() - est_pos
                if 25 > abs(dev[2]):
                    self.mode = 'Stabilize_VS'
                    my_print('Controller entering Stabilize_VS Mode at time %.3f' % time)
                    self.hor_hover_PID_cntrl_x.reset()
                    self.hor_hover_PID_cntrl_y.reset()
                    self.vert_pos_PID_hover_cntrl.reset()
                    return self.compute(est_pos, est_vel, est_accel, est_wind_spd, time, drag_model, rpy)
                net_hor_hover_force_x, net_hor_hover_force_y = self.hor_hover_PID(dev[0], dev[1])
                net_vertical_force = self.vert_rate_PID(self.flight_plan.current_leg.climb_rate - est_vel[2])
                net_force = np.array([net_hor_hover_force_x, net_hor_hover_force_y, net_vertical_force])


            elif self.mode == 'Stabilize_VS':
                # Check if AC is near its target waypoint. If yes, change to next leg.
                dev = self.flight_plan.current_leg.get_target_pos() - est_pos
                hor_dev_sq = dev[0]**2 + dev[1]**2
                if 25 > abs(dev[2]):
                    self.mode = 'Stabilize'
                    my_print('Controller entering Stabilize Mode at time %.3f' % time)
                    self.hor_hover_PID_cntrl_x.reset()
                    self.hor_hover_PID_cntrl_y.reset()
                    self.vert_pos_PID_hover_cntrl.reset()
                    return self.compute(est_pos, est_vel, est_accel, est_wind_spd, time, drag_model, rpy)
                net_hor_hover_force_x, net_hor_hover_force_y = self.hor_hover_PID(dev[0], dev[1])
                net_vertical_force = self.vert_rate_PID(np.sign(self.flight_plan.current_leg.hdg[2]) * 2.5 - est_vel[2])
                net_force = np.array([net_hor_hover_force_x, net_hor_hover_force_y, net_vertical_force])

            elif self.mode == 'Spline_3D':
                # Estimate spline param of AC using est. pos and param search
                # search_vel = max(np.linalg.norm(est_vel), 5)
                search_vel = max(self.flight_plan.current_leg.tgt_speed, 5)
                new_ds = search_vel * self.interval / self.flight_plan.current_leg.spl_path_len
                new_ds = max(new_ds, self.flight_plan.current_leg.spl_ds)
                search_start = max(self.spl_param - new_ds * 5, 0)
                search_end = min(1 + new_ds, search_start + 50 * new_ds)
                search_space = np.arange(search_start,
                                         search_end + self.flight_plan.current_leg.spl_ds,
                                         # new_ds,
                                         self.flight_plan.current_leg.spl_ds)
                tgt_x, tgt_y, tgt_z, self.spl_param = param_search(self.flight_plan.current_leg.spl_func,
                                                                   search_space, est_pos, est_sigma=self.spl_param)
                self.spl_params.append(self.spl_param)
                self.spl_times.append(time)
                if self.spl_param >= self.flight_plan.current_leg.spl_end_param:
                    self.flight_plan.change_flight_leg(time)
                    self.mode = self.flight_plan.current_leg.get_mode()
                    if self.mode == 'Spline_3Df':
                        self.spl_param_counter = 0
                    self.hor_rate_pos_PID_cntrl.reset()
                    self.hor_cross_trk_pos_PID_cntrl.reset()
                    self.vert_rate_PID_cntrl.reset()
                    self.vert_pos_PID_moving_cntrl.reset()
                    return self.compute(est_pos, est_vel, est_accel, est_wind_spd, time, drag_model, rpy)

                #TO-DO: Calculate the net_force
                sigma = np.array(interpolate.splev(self.spl_param, self.flight_plan.current_leg.spl_func))

                dx_dy_dz = np.array(interpolate.splev(self.spl_param, self.flight_plan.current_leg.spl_func, der=1))
                d2x_d2y_d2z = np.array(interpolate.splev(self.spl_param, self.flight_plan.current_leg.spl_func, der=2))
                d3x_d3y_d3z = np.array(interpolate.splev(self.spl_param, self.flight_plan.current_leg.spl_func, der=3))
                d4x_d4y_d4z = np.array(interpolate.splev(self.spl_param, self.flight_plan.current_leg.spl_func, der=4))
                s1_s1 = np.dot(dx_dy_dz, dx_dy_dz)
                s2_s2 = np.dot(d2x_d2y_d2z, d2x_d2y_d2z)

                s1_s2 = np.dot(dx_dy_dz, d2x_d2y_d2z)
                s1_s3 = np.dot(dx_dy_dz, d3x_d3y_d3z)

                s2_s3 = np.dot(d2x_d2y_d2z, d3x_d3y_d3z)

                etp = s1_s1 ** (-1.5) * (d2x_d2y_d2z * s1_s1 - dx_dy_dz * (s1_s2))
                et = dx_dy_dz / np.sqrt(s1_s1)
                ep = etp / np.sqrt(np.dot(etp, etp))
                eb = np.cross(et.T, ep.T).T

                along_trk_dev = self.flight_plan.current_leg.tgt_speed - est_vel @ et
                cross_trk_dev = (sigma - est_pos) @ ep
                vert_dev = (sigma - est_pos) @ eb

                dev = np.array([along_trk_dev, cross_trk_dev, vert_dev])

                # est_vel = et

                k_mat = np.eye(3) * s1_s1 ** (-1.5) * (s1_s1 *
                                                       s2_s2 -
                                                       s1_s2 ** 2) ** 0.5

                tau_mat = 1 / (s1_s1 * s2_s2 -
                               s1_s2 ** 2) * np.array([dx_dy_dz, d2x_d2y_d2z, d3x_d3y_d3z]).T

                d_k_mat = ((((s1_s1 * s2_s2 - s1_s2 ** 2) ** (-0.5) *
                             (s1_s1 * s2_s3 -
                              s1_s2 * s1_s3)) * np.eye(3) -
                            3 * s1_s1 ** 0.5 * k_mat * s1_s2) *
                           s1_s1 ** (-1.5))

                new_d_tau = (1/(s1_s1 * s2_s2 - s1_s2**2) * np.array([d2x_d2y_d2z, d3x_d3y_d3z, d4x_d4y_d4z]).T -
                             (s1_s1 * s2_s2 - s1_s2**2)**(-2) * 2 * (s1_s2 * s2_s3 - s1_s2 * s1_s3 *
                                                                     np.array([dx_dy_dz, d2x_d2y_d2z, d3x_d3y_d3z]).T)
                            )

                tpb = np.concatenate([et, ep, eb])
                super_mat = np.array([[np.zeros((3, 3)), k_mat, np.zeros((3, 3))],
                                      [-k_mat, np.zeros((3, 3)), tau_mat],
                                      [np.zeros((3, 3)), -tau_mat, np.zeros((3, 3))]]).transpose(0, 2, 1, 3).reshape(9,
                                                                                                                     9)
                d_super_mat = np.array([[np.zeros((3, 3)), d_k_mat, np.zeros((3, 3))],
                                        [-d_k_mat, np.zeros((3, 3)), new_d_tau],
                                        [np.zeros((3, 3)), -new_d_tau, np.zeros((3, 3))]]).transpose(0, 2, 1,
                                                                                                     3).reshape(9, 9)
                d_tpb = (s1_s1 ** 0.5 *
                         super_mat @ tpb)
                d2_tpb = (s1_s1 ** (-0.5) * s1_s2 * super_mat +
                          s1_s1 ** (0.5) * d_super_mat +
                          s1_s1 * (super_mat @ super_mat)) @ tpb
                d_t, d_p, d_b = d_tpb.reshape(3, 3)
                d2_t, d2_p, d2_b = d2_tpb.reshape(3, 3)

                # Careful of array sizing here; est_vel is almost always just a (3,) array...
                eta2 = np.dot(est_vel, et)  # tangential speed
                alpha1 = eta2 * s1_s1 ** (-0.5) * np.dot(d_t, est_vel)

                alpha2 = (eta2 * s1_s1 ** (-0.5) * np.dot(d_p, 2 * est_vel - eta2 * et) +
                          np.dot(est_pos - sigma, (d2_p * (eta2 ** 2) / s1_s1 +
                                                   d_p * (alpha1 / np.sqrt(s1_s1) -
                                                          eta2 ** 2 * s1_s2 / s1_s1 ** 2))
                                 )
                          )
                alpha3 = (eta2 * s1_s1 ** (-0.5) * np.dot(d_b, 2 * est_vel - eta2 * et) +
                          np.dot(est_pos - sigma,
                                 (d2_b * (eta2 ** 2) / s1_s1 +
                                  d_b * (alpha1 / np.sqrt(s1_s1) - eta2 ** 2 * s1_s2 / (s1_s1 ** 2)))
                                 )
                          )

                beta1 = et
                beta2 = ep + np.dot(est_pos - sigma, d_p) * (1 / np.sqrt(s1_s1)) * beta1
                beta3 = eb + np.dot(est_pos - sigma, d_b) * beta1 / np.sqrt(s1_s1)
                beta_mat = np.array([beta1, beta2, beta3])

                # u is the auxiliary input in the FS frame coordinates
                u = np.array([self.hor_rate_pos_PID(along_trk_dev),
                              self.spline_cross_trk_PID(cross_trk_dev),
                              # self.hor_cross_trk_pos_PID(cross_trk_dev),
                              self.vert_pos_PID_hover(vert_dev)])

                # Put everything together
                net_force = np.linalg.inv(beta_mat) @ (u - np.array([alpha1, alpha2, alpha3]))


            elif self.mode == 'Spline_3Df':
                # Estimate spline param of AC using est. pos and param search

                search_vel = max(self.flight_plan.current_leg.tgt_speed, 5)
                new_ds = search_vel * self.interval / self.flight_plan.current_leg.spl_path_len
                new_ds = max(new_ds, self.flight_plan.current_leg.spl_ds)
                search_start = max(self.spl_param - new_ds * 5, 0)
                search_end = min(1 + new_ds, search_start + 50 * new_ds)
                search_space = np.arange(search_start,
                                         search_end + self.flight_plan.current_leg.spl_ds,
                                         # new_ds,
                                         self.flight_plan.current_leg.spl_ds)
                # except:
                #     print('search_start is {a}, search_end is {b}, spl_param is {c}'.
                #           format(a=search_start, b=search_end, c=self.spl_param))
                tgt_x, tgt_y, tgt_z,  spl_param = param_search(self.flight_plan.current_leg.spl_func,
                                                                    search_space, est_pos, est_sigma=self.spl_param)
                if abs(spl_param - self.spl_param)/new_ds < 0.001:
                    self.spl_param_counter += 1
                else:
                    self.spl_param_counter = 0
                self.spl_param = spl_param
                dist_to_end_wpt = np.linalg.norm(est_pos - self.flight_plan.current_leg.get_target_pos())
                self.spl_params.append(self.spl_param)
                self.spl_times.append(time)
                if (self.spl_param >= self.flight_plan.current_leg.spl_end_param) | (dist_to_end_wpt <= 10) | \
                        (self.spl_param_counter == 100):
                    if self.flight_plan.change_flight_leg(time) == 'TERMINATE FLIGHT':
                        return self.compute(est_pos, est_vel, est_accel, est_wind_spd, time, drag_model, rpy)
                    self.mode = self.flight_plan.current_leg.get_mode()
                    self.hor_rate_pos_PID_cntrl.reset()
                    self.hor_cross_trk_pos_PID_cntrl.reset()
                    self.vert_rate_PID_cntrl.reset()
                    self.vert_pos_PID_moving_cntrl.reset()
                    self.spl_param = 0

                #TO-DO: Calculate the net_force
                sigma = np.array(interpolate.splev(self.spl_param, self.flight_plan.current_leg.spl_func))

                dx_dy_dz = np.array(interpolate.splev(self.spl_param, self.flight_plan.current_leg.spl_func, der=1))
                d2x_d2y_d2z = np.array(interpolate.splev(self.spl_param, self.flight_plan.current_leg.spl_func, der=2))
                d3x_d3y_d3z = np.array(interpolate.splev(self.spl_param, self.flight_plan.current_leg.spl_func, der=3))
                d4x_d4y_d4z = np.array(interpolate.splev(self.spl_param, self.flight_plan.current_leg.spl_func, der=4))
                s1_s1 = np.dot(dx_dy_dz, dx_dy_dz)
                s2_s2 = np.dot(d2x_d2y_d2z, d2x_d2y_d2z)

                s1_s2 = np.dot(dx_dy_dz, d2x_d2y_d2z)
                s1_s3 = np.dot(dx_dy_dz, d3x_d3y_d3z)

                s2_s3 = np.dot(d2x_d2y_d2z, d3x_d3y_d3z)

                etp = s1_s1 ** (-1.5) * (d2x_d2y_d2z * s1_s1 - dx_dy_dz * (s1_s2))
                et = dx_dy_dz / np.sqrt(s1_s1)
                ep = etp / np.sqrt(np.dot(etp, etp))
                eb = np.cross(et.T, ep.T).T

                along_trk_dev = self.flight_plan.current_leg.tgt_speed - est_vel @ et
                cross_trk_dev = (sigma - est_pos) @ ep
                vert_dev = (sigma - est_pos) @ eb

                dev = np.array([along_trk_dev, cross_trk_dev, vert_dev])

                # est_vel = et

                k_mat = np.eye(3) * s1_s1 ** (-1.5) * (s1_s1 *
                                                       s2_s2 -
                                                       s1_s2 ** 2) ** 0.5

                tau_mat = 1 / (s1_s1 * s2_s2 -
                               s1_s2 ** 2) * np.array([dx_dy_dz, d2x_d2y_d2z, d3x_d3y_d3z]).T

                d_k_mat = ((((s1_s1 * s2_s2 - s1_s2 ** 2) ** (-0.5) *
                             (s1_s1 * s2_s3 -
                              s1_s2 * s1_s3)) * np.eye(3) -
                            3 * s1_s1 ** 0.5 * k_mat * s1_s2) *
                           s1_s1 ** (-1.5))

                new_d_tau = (1 / (s1_s1 * s2_s2 - s1_s2 ** 2) * np.array([d2x_d2y_d2z, d3x_d3y_d3z, d4x_d4y_d4z]).T -
                             (s1_s1 * s2_s2 - s1_s2 ** 2) ** (-2) * 2 * (s1_s2 * s2_s3 - s1_s2 * s1_s3 *
                                                                         np.array(
                                                                             [dx_dy_dz, d2x_d2y_d2z, d3x_d3y_d3z]).T)
                             )

                tpb = np.concatenate([et, ep, eb])
                super_mat = np.array([[np.zeros((3, 3)), k_mat, np.zeros((3, 3))],
                                      [-k_mat, np.zeros((3, 3)), tau_mat],
                                      [np.zeros((3, 3)), -tau_mat, np.zeros((3, 3))]]).transpose(0, 2, 1, 3).reshape(9,
                                                                                                                     9)
                d_super_mat = np.array([[np.zeros((3, 3)), d_k_mat, np.zeros((3, 3))],
                                        [-d_k_mat, np.zeros((3, 3)), new_d_tau],
                                        [np.zeros((3, 3)), -new_d_tau, np.zeros((3, 3))]]).transpose(0, 2, 1,
                                                                                                     3).reshape(9, 9)
                d_tpb = (s1_s1 ** 0.5 *
                         super_mat @ tpb)
                d2_tpb = (s1_s1 ** (-0.5) * s1_s2 * super_mat +
                          s1_s1 ** (0.5) * d_super_mat +
                          s1_s1 * (super_mat @ super_mat)) @ tpb
                d_t, d_p, d_b = d_tpb.reshape(3, 3)
                d2_t, d2_p, d2_b = d2_tpb.reshape(3, 3)

                # Careful of array sizing here; est_vel is almost always just a (3,) array...
                eta2 = np.dot(est_vel, et)  # tangential speed
                alpha1 = eta2 * s1_s1 ** (-0.5) * np.dot(d_t, est_vel)

                alpha2 = (eta2 * s1_s1 ** (-0.5) * np.dot(d_p, 2 * est_vel - eta2 * et) +
                          np.dot(est_pos - sigma, (d2_p * (eta2 ** 2) / s1_s1 +
                                                   d_p * (alpha1 / np.sqrt(s1_s1) -
                                                          eta2 ** 2 * s1_s2 / s1_s1 ** 2))
                                 )
                          )
                alpha3 = (eta2 * s1_s1 ** (-0.5) * np.dot(d_b, 2 * est_vel - eta2 * et) +
                          np.dot(est_pos - sigma,
                                 (d2_b * (eta2 ** 2) / s1_s1 +
                                  d_b * (alpha1 / np.sqrt(s1_s1) - eta2 ** 2 * s1_s2 / (s1_s1 ** 2)))
                                 )
                          )

                beta1 = et
                beta2 = ep + np.dot(est_pos - sigma, d_p) * (1 / np.sqrt(s1_s1)) * beta1
                beta3 = eb + np.dot(est_pos - sigma, d_b) * beta1 / np.sqrt(s1_s1)
                beta_mat = np.array([beta1, beta2, beta3])

                # u is the auxiliary input in the FS frame coordinates
                u = np.array([self.hor_rate_pos_PID(along_trk_dev),
                              self.spline_cross_trk_PID(cross_trk_dev),
                              # self.hor_cross_trk_pos_PID(cross_trk_dev),
                              self.vert_pos_PID_hover(vert_dev)])

                # Put everything together
                net_force = np.linalg.inv(beta_mat) @ (u - np.array([alpha1, alpha2, alpha3]))



            net_force = net_force * self.aircraft_type.mass
            # # my_print('Net force is: ', net_force)

            # Now calculate thrust/force required by factoring drag and gravity
            # Assumes controller has some way of estimating or correcting for drag/gravity
            target_force = net_force - drag_model.get_drag(est_vel + est_wind_spd, rpy) + \
                           self.aircraft_type.mass * np.array([0, 0, 9.81])

            # Force the commanded "z" component of target_force to be greater than (-1g * mass)
            target_force[2] = np.clip(target_force[2], 9.81 * self.aircraft_type.mass * 0.6,
                                      self.aircraft_type.max_thrust)

            # Force x and y components of target_force to be clipped based on max RP...
            # There is some error with this mtd of calculating...
            # horizontal_force = target_force[0]**2 + target_force[1]**2
            # if horizontal_force > (self.max_horizontal_force * 1)**2:
            #     my_print("Horizontal force reaching max at time %.3f" % time)
            #     horizontal_force_scale = ((self.max_horizontal_force * 1)**2)/horizontal_force
            #     target_force[0] = target_force[0] * horizontal_force_scale
            #     target_force[1] = target_force[1] * horizontal_force_scale

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

            # if abs(target_force[0]) >= self.aircraft_type.mass * 9.81 * np.tan(self.aircraft_type.max_roll_pitch):
            #     my_print('X force component saturated at %.3f sec, tgt pre-clip P: %.3f '%(self.time, target_rpy[1]))

            # Limit target RPYs based on control mode...
            if (self.mode == 'Climb') | (self.mode == 'Hover') | (self.mode == 'Stabilize_VS'):
                target_rpy[0] = np.clip(target_rpy[0],
                                        -np.pi/2.5,
                                        np.pi/2.5)
                target_rpy[1] = np.clip(target_rpy[1],
                                        -np.pi/2.5,
                                        np.pi/2.5)
            else:
                target_rpy[0] = np.clip(target_rpy[0],
                                        -self.aircraft_type.max_roll_pitch,
                                        self.aircraft_type.max_roll_pitch)
                target_rpy[1] = np.clip(target_rpy[1],
                                        -self.aircraft_type.max_roll_pitch,
                                        self.aircraft_type.max_roll_pitch)

            # Note: controller error and net force are used mainly for debugging. Can be deleted in the future. Rmb to
            # edit the "State.py" file as well
            self.target_force = target_force
            self.target_rpy = target_rpy
            self.controller_dev = dev
            self.net_force = net_force

        return self.target_force, self.target_rpy, self.controller_dev, self.net_force

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
        # ---- PID target torques ----
        target_angular_accel = np.multiply(self.p_att, rpy_e) \
                               + np.multiply(self.i_att, self.integral_rpy_e) \
                               + np.multiply(self.d_att, d_rpy_e)

        return target_angular_accel

class PID(object):
    """PID Controller Class"""
    def __init__(self, update_rate, P, I, D, phase_delay=0):
        # super().__init__(update_rate, start_time, phase_delay)
        self.update_rate = update_rate
        self.I_Dev = 0
        # self.D_Dev = 0
        self.Prev_Dev = None
        self.P = P
        self.I = I
        self.D = D
        self.output = 0
    def cntrl(self, dev):
        if isinstance(self.Prev_Dev, type(None)):
            self.output = self.P * dev
        else:
            self.output = self.P * dev + self.I * self.I_Dev /self.update_rate + self.D * \
                          (dev - self.Prev_Dev) * self.update_rate
        self.I_Dev += dev
        self.Prev_Dev = dev
        return self.output
    def reset(self):
        self.I_Dev = 0
        self.Prev_Dev = None
        self.output = 0