# -*- coding: utf-8 -*-
"""
@Time    : 03/7/2022 9:41 AM
@Author  : Zhi Hao
@FileName: ConflictDetector.py
@Description: File containing implementation for various conflict detection algorithms.
Includes:
NMAC detection: based on distance to aircraft Centre-of-Mass
WCV detection: based on ^^
WCV detection: based on modified tau criteria
@Package dependency:
"""
import numpy as np
from CrossPlatformDev import my_print
from Engine.GlobalClock import Agent
from Engine.State import State


def calculate_tau(AC1_Pos, AC2_Pos, AC1_Vel, AC2_Vel, DMOD):
    r_hor = (AC2_Pos - AC1_Pos).copy()
    r_hor[2] = 0
    hor_dist = np.linalg.norm(r_hor)
    if hor_dist <= DMOD:
        tau_mod_hor = 0
    else:
        r_dot_hor = (1 / hor_dist) * np.dot(r_hor, AC2_Vel - AC1_Vel)
        if r_dot_hor == 0:
            tau_mod_hor = 999
        else:
            tau_mod_hor = -(hor_dist ** 2 - DMOD ** 2) / (hor_dist * r_dot_hor)
    vert_dist = (AC2_Pos - AC1_Pos)[2]
    vert_speed = (AC2_Vel - AC1_Vel)[2]
    if vert_speed == 0:
        tau_vert = 999
    else:
        tau_vert = -vert_dist / vert_speed
    return tau_mod_hor, tau_vert


class ConflictDetector(Agent):
    """Class that helps to keep track of conflicts. Currently supports 2 main conflict detection algorithms:
    Distance-based criteria: determine if conflict exists based on position of 2 A/C
    Modified Tau criteria: determine if conflict exists based on position and relative speed (time-parametric criteria)
    After agent is initialized, it needs to have various conditions "added" to it, with user specified values for
    conflict radius/tau.
    During simulation, parameters should be updated using the various "update_xxx" methods.
    At the end of the simulation, list of conflict times can be retrieved using the appropriate conflict definition
    tag. There are also methods to help calculate the earliest conflict start time and conflict end time.
    """
    def __init__(self, update_rate, start_time, phase_delay=0, AC1_State=None, AC2_State=None, AC1_Controller=None):
        super().__init__(update_rate, start_time, phase_delay)
        self.conflict_definitions = {}
        self.r_hor_sq = None
        self.vert_dist = None
        self.tau_DMOD = {}
        self.taus = {}
        self.AC1_State = AC1_State
        self.AC2_State = AC2_State
        self.AC1_Controller = AC1_Controller

    def add_conflict_definition(self, condition_tag, condition):
        """Adds a conflict detection condition with a corresponding tag. Each condition should be a self-contained
        function that requires no inputs (necessary parameters should be bounded to this class as attributes) that
        evaluates to True iff the conflict definition/logic is satisfied by the A/C in question. """
        conflict_history = {'condition': condition, 'conflict_start_time': [], 'conflict_end_time': [],
                            'is_in_conflict': False, 'Rel_Hdg_(Desired_Track_Rad)': None, 'Rel_Hdg_(Actual_Rad)': None,
                            'Rel_Vel_Hdg_(Actual_Rad)': None,
                            'Rel_Hor_Dist': None, 'Rel_Vert_Dist': None,
                            'Rel_Hor_Dist_Delta':None, 'Rel_Vert_Dist_Delta':None
                            }
        self.conflict_definitions[condition_tag] = conflict_history

    def add_DMOD(self, DMOD_tag, DMOD):
        self.tau_DMOD[DMOD_tag] = DMOD
        if DMOD not in self.tau_DMOD.values():
            self.taus[DMOD_tag] = None

    def update_r_hor_sq_vert_dist(self, AC1_Pos, AC2_Pos):
        if not (isinstance(AC1_Pos, type(None)) | isinstance(AC2_Pos, type(None))):
            self.r_hor_sq = (AC1_Pos - AC2_Pos)[0] ** 2 + (AC1_Pos - AC2_Pos)[1] ** 2
            self.vert_dist = abs((AC1_Pos - AC2_Pos)[2])

    def update_taus(self, AC1_Pos, AC2_Pos, AC1_Vel, AC2_Vel):
        """Note: Make sure update_taus is called BEFORE condition is checked."""
        if not (isinstance(AC1_Pos, type(None)) | isinstance(AC2_Pos, type(None))):
            for DMOD_tag, DMOD in self.tau_DMOD.items():
                self.taus[DMOD_tag] = calculate_tau(AC1_Pos, AC2_Pos, AC1_Vel, AC2_Vel, DMOD)

    def gen_distance_condition(self, hor_dist_min, vert_dist_min):
        hor_r_min_sq = hor_dist_min ** 2

        def check():
            if not isinstance(self.r_hor_sq, type(None)):
                return (self.r_hor_sq <= hor_r_min_sq) & (self.vert_dist <= vert_dist_min)
            else:
                return False
        return check

    def gen_tau_mod_condition(self, tau_mod_min, tau_vert_min, vert_dist_min, DMOD):
        """Note: Make sure update_taus is called BEFORE condition is checked."""
        self.add_DMOD(str(DMOD), DMOD)

        def check():
            tau_mod, tau_vert = self.taus[str(DMOD)]
            return (0 <= tau_mod < tau_mod_min) & ((0 <= tau_vert < tau_vert_min) | (self.vert_dist <= vert_dist_min))
        return check

    def conflict_check(self, time):
        if super().check_time(time):
            # my_print('Time is %.1f, r_hor_sq is %.3f and vert_dist is %.3f' % (time, self.r_hor_sq, self.vert_dist))
            # my_print('Time is %.1f, taus: ' % (time), self.taus)
            for condition_history in self.conflict_definitions.values():
                if (not condition_history['is_in_conflict']) & condition_history['condition']():
                    condition_history['is_in_conflict'] = True
                    condition_history['conflict_start_time'].append(time)
                    if (not condition_history['Rel_Hdg_(Actual_Rad)']) & isinstance(self.AC1_State, State):
                        hor_vel_AC1 = self.AC1_State.gt_vel.copy()
                        hor_vel_AC1[2] = 0
                        norm_AC1_hor_vel = np.linalg.norm(hor_vel_AC1)
                        hor_vel_AC1 = hor_vel_AC1/norm_AC1_hor_vel
                        hor_vel_AC2 = self.AC2_State.gt_vel.copy()
                        hor_vel_AC2[2] = 0
                        norm_AC2_hor_vel = np.linalg.norm(hor_vel_AC2)
                        hor_vel_AC2 = hor_vel_AC2 / norm_AC2_hor_vel
                        uAC1_dot_uAC2 = np.dot(hor_vel_AC1, hor_vel_AC2)
                        # Add relative heading at conflict start time based on GT positions
                        condition_history['Rel_Vel_Hdg_(Actual_Rad)'] = np.arccos(uAC1_dot_uAC2)

                        # Add relative position at conflict start time based on GT positions
                        hor_pos_AC1 = self.AC1_State.gt_pos.copy()
                        hor_pos_AC2 = self.AC2_State.gt_pos.copy()
                        condition_history['Rel_Vert_Dist'] = (self.AC2_State.gt_pos - self.AC1_State.gt_pos)[2]
                        condition_history['Rel_Vert_Dist_Delta'] = abs((self.AC2_State.gt_pos -
                                                                        self.AC1_State.gt_pos)[2]) - abs(self.vert_dist)
                        hor_pos_AC1[2] = 0
                        hor_pos_AC2[2] = 0
                        rel_AC1_AC2_pos = hor_pos_AC2 - hor_pos_AC1
                        rel_AC1_AC2_pos_unit = rel_AC1_AC2_pos/np.linalg.norm(rel_AC1_AC2_pos)
                        condition_history['Rel_Hor_Dist'] = np.linalg.norm(rel_AC1_AC2_pos)
                        condition_history['Rel_Hor_Dist_Delta'] = np.linalg.norm(rel_AC1_AC2_pos) - \
                                                                  np.sqrt(self.r_hor_sq)
                        if not isinstance(self.AC1_Controller.tangential_unit_vector_planar, type(None)):
                            condition_history['Rel_Hdg_(Desired_Track_Rad)'] = np.arccos(np.dot(rel_AC1_AC2_pos_unit,
                                                                            self.AC1_Controller.tangential_unit_vector_planar))
                        condition_history['Rel_Hdg_(Actual_Rad)'] = np.arccos(np.dot(rel_AC1_AC2_pos_unit, hor_vel_AC1))

                elif condition_history['is_in_conflict'] & (not condition_history['condition']()):
                    condition_history['is_in_conflict'] = False
                    condition_history['conflict_end_time'].append(time)

    def update_and_conflict_check(self, time, AC1_Pos, AC2_Pos, AC1_Vel, AC2_Vel):
        if super().check_time(time):
            self.update_taus(AC1_Pos, AC2_Pos, AC1_Vel, AC2_Vel)
            self.update_r_hor_sq_vert_dist(AC1_Pos, AC2_Pos)
            # my_print('Time is %.1f, r_hor_sq is %.3f and vert_dist is %.3f' % (time, np.sqrt(self.r_hor_sq),
            #                                                                    self.vert_dist))
            # my_print('Time is %.1f, taus: ' % (time), self.taus)
            for condition_history in self.conflict_definitions.values():
                if (not condition_history['is_in_conflict']) & condition_history['condition']():
                    condition_history['is_in_conflict'] = True
                    condition_history['conflict_start_time'].append(time)
                    if (not condition_history['Rel_Hdg_(Actual_Rad)']) & isinstance(self.AC1_State, State):
                        hor_vel_AC1 = self.AC1_State.gt_vel.copy()
                        hor_vel_AC1[2] = 0
                        norm_AC1_hor_vel = np.linalg.norm(hor_vel_AC1)
                        hor_vel_AC1 = hor_vel_AC1/norm_AC1_hor_vel
                        hor_vel_AC2 = self.AC2_State.gt_vel.copy()
                        hor_vel_AC2[2] = 0
                        norm_AC2_hor_vel = np.linalg.norm(hor_vel_AC2)
                        hor_vel_AC2 = hor_vel_AC2 / norm_AC2_hor_vel
                        uAC1_dot_uAC2 = np.dot(hor_vel_AC1, hor_vel_AC2)
                        # Add relative heading at conflict start time based on GT positions
                        condition_history['Rel_Vel_Hdg_(Actual_Rad)'] = np.arccos(uAC1_dot_uAC2)

                        # Add relative position at conflict start time based on GT positions
                        hor_pos_AC1 = self.AC1_State.gt_pos.copy()
                        hor_pos_AC2 = self.AC2_State.gt_pos.copy()
                        condition_history['Rel_Vert_Dist'] = (self.AC2_State.gt_pos - self.AC1_State.gt_pos)[2]
                        condition_history['Rel_Vert_Dist_Delta'] = abs((self.AC2_State.gt_pos -
                                                                        self.AC1_State.gt_pos)[2]) - abs(self.vert_dist)
                        hor_pos_AC1[2] = 0
                        hor_pos_AC2[2] = 0
                        rel_AC1_AC2_pos = hor_pos_AC2 - hor_pos_AC1
                        rel_AC1_AC2_pos_unit = rel_AC1_AC2_pos/np.linalg.norm(rel_AC1_AC2_pos)
                        condition_history['Rel_Hor_Dist'] = np.linalg.norm(rel_AC1_AC2_pos)
                        condition_history['Rel_Hor_Dist_Delta'] = np.linalg.norm(rel_AC1_AC2_pos) - \
                                                                  np.sqrt(self.r_hor_sq)
                        if not isinstance(self.AC1_Controller.tangential_unit_vector_planar, type(None)):
                            condition_history['Rel_Hdg_(Desired_Track_Rad)'] = np.arccos(np.dot(rel_AC1_AC2_pos_unit,
                                                                            self.AC1_Controller.tangential_unit_vector_planar))
                        condition_history['Rel_Hdg_(Actual_Rad)'] = np.arccos(np.dot(rel_AC1_AC2_pos_unit, hor_vel_AC1))

                elif condition_history['is_in_conflict'] & (not condition_history['condition']()):
                    condition_history['is_in_conflict'] = False
                    condition_history['conflict_end_time'].append(time)


    def get_earliest_conflict_start_time(self, conflict_tag):
        if len(self.conflict_definitions[conflict_tag]['conflict_start_time']) > 0:
            return min(self.conflict_definitions[conflict_tag]['conflict_start_time'])
        else:
            return None

    def get_latest_conflict_end_time(self, conflict_tag):
        if len(self.conflict_definitions[conflict_tag]['conflict_end_time']) > 0:
            return max(self.conflict_definitions[conflict_tag]['conflict_end_time'])
        else:
            return None

    def history_of_conflict(self, conflict_tag):
        return len(self.conflict_definitions[conflict_tag]['conflict_start_time']) > 0



