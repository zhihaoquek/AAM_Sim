# -*- coding: utf-8 -*-
"""
@Time    : 02/11/2022 9:41 AM
@Author  : Zhi Hao
@FileName: FlightPlan.py
@Description: Details A/C FlightPlan.
@Package dependency:
"""
import numpy as np
from CrossPlatformDev import my_print
from Engine.GlobalClock import Agent
import pandas as pd


class FlightPlan(object):
    """Object containing information about the various flight legs in a mission.
    Note that flight legs start with number 1, NOT 0. """
    def __init__(self,
                 leg_spd,
                 mode,
                 wpt_start, wpt_start_time,
                 wpt_end, wpt_end_time, duration):
        self.plan = pd.DataFrame({'Leg No.': np.arange(len(wpt_start))+1, 'Target Speed': leg_spd, 'Mode': mode,
                                  'Starting Wpt': wpt_start, 'EDT': wpt_start_time,
                                  'Ending Wpt': wpt_end, 'ETA': wpt_end_time, 'Duration': duration})
        self.current_leg_num = 1
        self.current_leg = FlightLeg(self.plan.iloc[self.current_leg_num - 1]['Mode'],
                                     self.plan.iloc[self.current_leg_num - 1]['Starting Wpt'],
                                     self.plan.iloc[self.current_leg_num - 1]['EDT'],
                                     self.plan.iloc[self.current_leg_num - 1]['Ending Wpt'],
                                     self.plan.iloc[self.current_leg_num - 1]['ETA'],
                                     self.plan.iloc[self.current_leg_num - 1]['Target Speed']
                                     )

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
                                             self.plan.iloc[self.current_leg_num - 1]['Target Speed']
                                             )
            else:
                self.current_leg = FlightLeg(self.plan.iloc[self.current_leg_num - 1]['Mode'],
                                             self.plan.iloc[self.current_leg_num - 1]['Starting Wpt'],
                                             self.plan.iloc[self.current_leg_num - 1]['EDT'],
                                             self.plan.iloc[self.current_leg_num - 1]['Ending Wpt'],
                                             self.plan.iloc[self.current_leg_num - 1]['ETA'],
                                             self.plan.iloc[self.current_leg_num - 1]['Target Speed']
                                             )
            my_print('Next Wpt is: ', self.current_leg.target_pos)
            my_print('Hdg is: ', self.current_leg.hdg)
        elif self.current_leg_num == len(self.plan):
            self.current_leg = None
            return 'TERMINATE FLIGHT'


class FlightLeg(object):
    """Note: EDT/ETA not in use for point2point flight legs yet, for future development. """
    def __init__(self, mode, starting_pt, edt, ending_pt, eta, cruise_spd):
        self.mode = mode
        self.target_pos = ending_pt
        self.starting_pos = starting_pt
        self.tgt_speed = cruise_spd
        self.hdg = self.target_pos - self.starting_pos
        self.EDT = edt
        self.ETA = eta

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









