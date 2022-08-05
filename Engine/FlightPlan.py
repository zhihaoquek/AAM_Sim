# -*- coding: utf-8 -*-
"""
@Time    : 02/11/2022 9:41 AM
@Author  : Zhi Hao
@FileName: FlightPlan.py
@Description: Details A/C FlightPlan.
@Package dependency:
"""
import numpy as np
from CrossPlatformDev import my_print, param_search, auto_scale
from Engine.GlobalClock import Agent
import pandas as pd
from scipy import interpolate


class FlightPlan(object):
    """Object containing information about the various flight legs in a mission.
    Note that flight legs start with number 1, NOT 0. """
    def __init__(self,
                 leg_spd,
                 mode,
                 wpt_start, wpt_start_time,
                 wpt_end, wpt_end_time, duration, climb_rate):
        self.plan = pd.DataFrame({'Leg No.': np.arange(len(wpt_start))+1, 'Target Speed': leg_spd, 'Mode': mode,
                                  'Starting Wpt': wpt_start, 'EDT': wpt_start_time,
                                  'Ending Wpt': wpt_end, 'ETA': wpt_end_time, 'Duration': duration,
                                  'Climb Rate': climb_rate})
        self.current_leg_num = 1
        self.current_leg = FlightLeg(self.plan.iloc[self.current_leg_num - 1]['Mode'],
                                     self.plan.iloc[self.current_leg_num - 1]['Starting Wpt'],
                                     self.plan.iloc[self.current_leg_num - 1]['EDT'],
                                     self.plan.iloc[self.current_leg_num - 1]['Ending Wpt'],
                                     self.plan.iloc[self.current_leg_num - 1]['ETA'],
                                     self.plan.iloc[self.current_leg_num - 1]['Target Speed'],
                                     self.plan.iloc[self.current_leg_num - 1]['Climb Rate']
                                     )
        counter = 0
        # zipped_fp = zip(self.plan['Leg No.'], self.plan['Mode'])
        spl_seg = []
        spl_num = []
        spl_curr = 0
        spl_num_df_list = []
        spl_summary = []
        while counter < len(self.plan):
            # print(counter)
            # zipped_fp[1]
            if self.plan.iloc[counter]['Mode'] == 'Spline_3D':
                # Check how many consecutive waypoints are in 'Spline_3D' mode
                spl_num_df_list.append(spl_curr)
                clone_cnt = counter + 1
                approx_len = np.linalg.norm(self.plan.iloc[counter]['Ending Wpt'] -
                                            self.plan.iloc[counter]['Starting Wpt'])
                while clone_cnt < len(self.plan):
                    if self.plan.iloc[clone_cnt]['Mode'] == 'Spline_3D':
                        spl_num_df_list.append(spl_curr)
                        approx_len += np.linalg.norm(self.plan.iloc[counter]['Ending Wpt'] -
                                                     self.plan.iloc[counter]['Starting Wpt'])
                        clone_cnt += 1
                    else:
                        break
                    final_cnt = clone_cnt - 1
                spl_num.append(spl_curr)

                # Get waypoints that define spline segment
                wpts = np.array([self.plan.iloc[counter]['Starting Wpt']])
                clone_clone_cnt = counter
                while clone_clone_cnt <= final_cnt:
                    wpts = np.append(wpts, [self.plan.iloc[clone_clone_cnt]['Ending Wpt']], axis=0)
                    clone_clone_cnt += 1
                # spl_seg, spl_s = interpolate.splprep([x_coords, y_coords], s=0)
                spl_curr += 1
                func, u = interpolate.splprep([wpts.T[0], wpts.T[1], wpts.T[2]], s=0, k=4)
                spl_seg.append(func)
                spl_summary.append(auto_scale(func, 0.1, approx_len, 0.01, 0.5))
                counter = final_cnt + 1
            else:
                counter += 1
                spl_num_df_list.append('NA')

        self.spl_df = pd.DataFrame({'Spline Func': spl_seg, 'Spline Number': spl_num, 'Spline Summary': spl_summary})
        self.plan['Spline Number'] = spl_num_df_list

        final_spline_leg_numbers = []
        for i in self.plan['Spline Number'].unique():
            if type(i) != str:
                # print(i)
                test_df = self.plan[self.plan['Spline Number'] == i]
                # print(test_df)
                print('Max Leg No. for Spline Number %.0f is %.0f' % (i, test_df['Leg No.'].max()))
                final_spline_leg_numbers.append(test_df['Leg No.'].max())

        def rename(a, b):
            if ((a == 'Spline_3D') & (b in final_spline_leg_numbers)):
                return a + 'f'
            else:
                return a

        self.plan['Mode'] = [rename(*a) for a in tuple(zip(self.plan['Mode'], self.plan['Leg No.']))]

        ending_params = []
        leg_nums = []
        search_space_start = 0
        search_space_end = 1
        est_sigma = 0
        for func, num, summary in zip(spl_seg, spl_num, spl_summary):
            df = self.plan[self.plan['Spline Number'] == num]
            for mode, end_wpt, leg_num in zip(df['Mode'], df['Ending Wpt'], df['Leg No.']):
                # print(leg_num)
                if mode == 'Spline_3D':
                    ds = summary['ds_target_res']
                    search_space = np.arange(search_space_start, search_space_end + ds, ds)
                    xt, yt, zt = end_wpt
                    x, y, z, param = param_search(func, search_space, end_wpt, est_sigma)
                    # my_print('x_est %.1f, x_true %.1f, y_est %.1f, y_true %.1f' % (x, xt, y, yt))
                    ending_params.append(param)
                    search_space_start = param
                    # search_space_end = 1
                    est_sigma = param
                    leg_nums.append(leg_num)
                elif mode == 'Spline_3Df':
                    param = 1
                    ending_params.append(param)
                    search_space_start = 0
                    leg_nums.append(leg_num)
        df = pd.DataFrame({'Leg No.': leg_nums, 'Est. End. Sigma': ending_params})
        self.plan = pd.merge(self.plan, df, on='Leg No.', how='outer')
        if ((self.current_leg.mode == 'Spline_3D') | (self.current_leg.mode == 'Spline_3Df')):
            self.init_spline_flight_leg()

    def init_spline_flight_leg(self):
        self.current_leg.spl_num = self.plan[self.plan['Leg No.'] ==
                                             self.current_leg_num].iloc[0]['Spline Number']
        self.current_leg.spl_func = self.spl_df[self.spl_df['Spline Number'] ==
                                                self.current_leg.spl_num].iloc[0]['Spline Func']
        self.current_leg.spl_ds = self.spl_df[self.spl_df['Spline Number'] ==
                                              self.current_leg.spl_num].iloc[0]['Spline Summary']['ds_target_res']
        self.current_leg.spl_end_param = self.plan[self.plan['Leg No.'] ==
                                                   self.current_leg_num].iloc[0]['Est. End. Sigma']
        self.current_leg.spl_path_len = self.spl_df[self.spl_df['Spline Number'] ==
                                                    self.current_leg.spl_num].iloc[0]['Spline Summary']['Path_len']

    def change_flight_leg(self, time, override_eta=True):
        my_print('CHANGE FLIGHT LEG')
        if self.current_leg_num < len(self.plan):
            self.current_leg_num += 1
            # Override EDT/ETA based on init time and "duration"
            if override_eta:
                self.current_leg = FlightLeg(self.plan.iloc[self.current_leg_num - 1]['Mode'],
                                             self.plan.iloc[self.current_leg_num - 1]['Starting Wpt'],
                                             time,
                                             self.plan.iloc[self.current_leg_num - 1]['Ending Wpt'],
                                             time + self.plan.iloc[self.current_leg_num - 1]['Duration'],
                                             self.plan.iloc[self.current_leg_num - 1]['Target Speed'],
                                             self.plan.iloc[self.current_leg_num - 1]['Climb Rate']
                                             )
            else:
                self.current_leg = FlightLeg(self.plan.iloc[self.current_leg_num - 1]['Mode'],
                                             self.plan.iloc[self.current_leg_num - 1]['Starting Wpt'],
                                             self.plan.iloc[self.current_leg_num - 1]['EDT'],
                                             self.plan.iloc[self.current_leg_num - 1]['Ending Wpt'],
                                             self.plan.iloc[self.current_leg_num - 1]['ETA'],
                                             self.plan.iloc[self.current_leg_num - 1]['Target Speed'],
                                             self.plan.iloc[self.current_leg_num - 1]['Climb Rate']
                                             )
            if ((self.current_leg.mode == 'Spline_3D') | (self.current_leg.mode == 'Spline_3Df')):
                self.init_spline_flight_leg()
            my_print('Next Wpt is: ', self.current_leg.target_pos)
            my_print('Hdg is: ', self.current_leg.hdg)
        elif self.current_leg_num == len(self.plan):
            self.current_leg = None
            return 'TERMINATE FLIGHT'


class FlightLeg(object):
    """Note: EDT/ETA not in use for point2point flight legs yet, for future development. """
    def __init__(self, mode, starting_pt, edt, ending_pt, eta, cruise_spd, climb_rate):
        self.mode = mode
        self.target_pos = ending_pt
        self.starting_pos = starting_pt
        self.tgt_speed = cruise_spd
        self.hdg = self.target_pos - self.starting_pos
        if np.linalg.norm(self.hdg) == 0:
            self.hdg_normed = np.zeros(3)
        else:
            self.hdg_normed = self.hdg / np.linalg.norm(self.hdg)
        self.EDT = edt
        self.ETA = eta
        if self.mode == 'Direct_P2P':
            self.cross_trk = np.cross(self.hdg, np.array([0, 0, 1])) / np.linalg.norm(self.hdg)
            along_trk = self.hdg.copy()
            along_trk[2] = 0
            self.along_trk = along_trk / np.linalg.norm(along_trk)
        if mode == 'Climb':
            self.climb_rate = climb_rate


    def get_target_pos(self, *parameter):
        # if self.mode == 'Hover':
        #     return self.target_pos
        return self.target_pos

    def get_eta(self):
        return self.ETA

    def get_mode(self):
        return self.mode

    def line_gen(self, param):
        return self.starting_pos + param * self.hdg

    def lambda_calculator(self, position):
        """A flight leg can be parameterized by lambda parameter.
        Lambda = 0 --> tangential A/C coordinate at start pt.
        Lambda = 1 --> tangential A/C coordinate at end pt.
        This function calculates lambda using A/C estimated position,
        i.e. where the A/C is along the flight leg.
        """
        CA = position - self.starting_pos
        return np.dot(CA, self.hdg)/np.dot(self.hdg, self.hdg)









