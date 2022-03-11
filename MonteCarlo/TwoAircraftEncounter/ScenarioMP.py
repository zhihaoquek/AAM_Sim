# -*- coding: utf-8 -*-
"""
@Time    : 03/7/2022 9:41 AM
@Author  : Zhi Hao
@FileName: ScenarioMP.py
@Description: Script for reading input parameters and running MultiProcessing Monte Carlo Simulation.
@Package dependency:
"""
import numpy as np
import os
import pandas as pd
from CrossPlatformDev import my_print, join_str
from Engine.GlobalClock import GlobalClock, Agent
from Engine.State import State
from Engine.Aircraft import AircraftType
from Engine.FlightPlan import FlightPlan
from Engine.MultiRotorController import MultiRotorController
from Engine.Wind import WindField
from Engine.Sensors import NavUpdate, GPSPosNavUpdate
from Engine.DragModel import DragModel
from Engine.TrackingSystem import *
from Engine.ConflictDetector import ConflictDetector

Init_Param_Path = join_str(os.getcwd(), 'MonteCarlo', 'TwoAircraftEncounter', 'Init_Param.csv')


def get_run(df, run):
    return df.loc[df['Run'] == run]


def get_val(df, key):
    return df.iloc[0][key]


def get_param():
    """For debugging use. To check if directory is correctly set to read the param file."""
    return pd.read_csv(Init_Param_Path)


def special_concat(trajectory):
    return np.concatenate(trajectory).reshape(len(trajectory), 8) # 8 is the length of each mini array


def extract_from_df(df, key, idx):
    """Extracts single column from df cells with 3D point values. """
    df_len = len(df)
    return np.concatenate(df[key].to_numpy()).reshape(df_len, 3)[:,idx]


def FP_gen(HDG, speed, time, num_wpts, ini_pos):
    """Function that generates a flight plan according to encounter design.
    A/C have fixed heading and will fly toward WPTs.
    Trajectory is such that there is a crossing point at the origin.
    HDG --> desired A/C HDG (counter-clockwise from x-axis in RADIANS)
    speed --> desired A/C cruise spd
    time --> approximate duration for each flight leg (between 2 consecutive wpts)
    num_wpts --> number of wpts along the flight path
    ini_pos --> initial position of the A/C
    """
    wpt_list = []
    wpt_times = []
    spds = []
    mode_list = []
    duration_list = []
    for i in range(num_wpts):
        wpt_list.append(ini_pos + i*speed*time*np.array([np.cos(HDG), np.sin(HDG), 0]))
        wpt_times.append(i*time*1.1) # Give a 10% margin to each leg duration
        if i > 0:
            spds.append(speed)
            mode_list.append('Direct_P2P')
            duration_list.append(time*1.1) # Give a 10% margin to each leg duration
    wpt_start_list = wpt_list[0:num_wpts-1]
    wpt_end_list = wpt_list[1:]
    wpt_start_time = wpt_times[0:num_wpts-1]
    wpt_end_time = wpt_times[1:]
    return (FlightPlan(spds, mode_list,
                       wpt_start_list, wpt_start_time,
                       wpt_end_list, wpt_end_time, duration_list), wpt_list)


def simulate_encounter(run):
    return simulate_encounter_gen(run, debug=False)


def simulate_encounter_debug(run):
    return simulate_encounter_gen(run, debug=True)


def simulate_encounter_gen(run, debug):
    params = pd.read_csv(Init_Param_Path)
    run_params = get_run(params, run)

    # Global Params

    GlobalPhysicsUpdateRate = get_val(run_params, 'GlobalPhysicsUpdateRate')
    GlobalSimStopTime = get_val(run_params, 'GlobalSimStopTime')
    Cruise_Leg_Time = get_val(run_params, 'Cruise_Leg_Time')
    Num_Legs = int(get_val(run_params, 'Num_Legs'))

    WindUpdateRate = get_val(run_params, 'WindUpdateRate')
    Wind_auto_x = get_val(run_params, 'Wind_auto_x')
    Wind_auto_y = get_val(run_params, 'Wind_auto_y')
    Wind_auto_z = get_val(run_params, 'Wind_auto_z')
    Wind_std_x = get_val(run_params, 'Wind_std_x')
    Wind_std_y = get_val(run_params, 'Wind_std_y')
    Wind_std_z = get_val(run_params, 'Wind_std_z')
    Wind_mean_x = get_val(run_params, 'Wind_mean_x')
    Wind_mean_y = get_val(run_params, 'Wind_mean_y')
    Wind_mean_z = get_val(run_params, 'Wind_mean_z')

    # AC1 Params

    AC1_Prop_Diameter = get_val(run_params, 'AC1_Prop_Diameter')
    AC1_Mass = get_val(run_params, 'AC1_Mass')
    AC1_HDG = get_val(run_params, 'AC1_HDG')
    AC1_Cruise_Speed = get_val(run_params, 'AC1_Cruise_Speed')
    AC1_Ini_Pos = np.array([get_val(run_params, 'AC1_Ini_Pos_x'),
                            get_val(run_params, 'AC1_Ini_Pos_y'),
                            get_val(run_params, 'AC1_Ini_Pos_z')])
    AC1_Ini_Vel = np.array([get_val(run_params, 'AC1_Ini_Vel_x'),
                            get_val(run_params, 'AC1_Ini_Vel_y'),
                            get_val(run_params, 'AC1_Ini_Vel_z')])

    AC1_GPS_horizontal_std = get_val(run_params, 'AC1_GPS_horizontal_std')
    AC1_GPS_horizontal_auto = get_val(run_params, 'AC1_GPS_horizontal_auto')
    AC1_GPS_vert_std = get_val(run_params, 'AC1_GPS_vert_std')
    AC1_GPS_vert_auto = get_val(run_params, 'AC1_GPS_vert_auto')

    AC1_Ini_Pos_Err = np.array([get_val(run_params, 'AC1_Ini_Pos_Err_x'),
                                get_val(run_params, 'AC1_Ini_Pos_Err_y'),
                                get_val(run_params, 'AC1_Ini_Pos_Err_z')])

    AC1_Ini_Vel_Err = np.array([get_val(run_params, 'AC1_Ini_Vel_Err_x'),
                                get_val(run_params, 'AC1_Ini_Vel_Err_y'),
                                get_val(run_params, 'AC1_Ini_Vel_Err_z')])

    AC1_Ini_RPY = np.array([get_val(run_params, 'AC1_Ini_RPY_r'),
                            get_val(run_params, 'AC1_Ini_RPY_p'),
                            get_val(run_params, 'AC1_Ini_RPY_y')])

    AC1_PhysicsUpdateRate = get_val(run_params, 'AC1_PhysicsUpdateRate')
    AC1_Controller_UpdateRate = get_val(run_params, 'AC1_Controller_UpdateRate')
    AC1_Start_Time = get_val(run_params, 'AC1_Start_Time')

    AC1_PosUpdateRate = get_val(run_params, 'AC1_PosUpdateRate')
    AC1_VelUpdateRate = get_val(run_params, 'AC1_VelUpdateRate')
    AC1_AccelUpdateRate = get_val(run_params, 'AC1_AccelUpdateRate')
    AC1_AirSpdSensorUpdateRate = get_val(run_params, 'AC1_AirSpdSensorUpdateRate')

    # Initialize AC1 Tracking System

    AC1_TrackingUnit_UpdateRate = get_val(run_params, 'AC1_TrackingUnit_UpdateRate')
    AC1_TrackingUnit_Pos_Quant = int(get_val(run_params, 'AC1_TrackingUnit_Pos_Quant'))
    AC1_TrackingUnit_Vel_Quant = int(get_val(run_params, 'AC1_TrackingUnit_Vel_Quant'))
    AC1_TrackingUnit_Avail = get_val(run_params, 'AC1_TrackingUnit_Avail')
    AC1_Latency = get_val(run_params, 'AC1_Latency')

    # AC2 Params

    AC2_Prop_Diameter = get_val(run_params, 'AC2_Prop_Diameter')
    AC2_Mass = get_val(run_params, 'AC2_Mass')
    AC2_HDG = get_val(run_params, 'AC2_HDG')
    AC2_Cruise_Speed = get_val(run_params, 'AC2_Cruise_Speed')
    AC2_Ini_Pos = np.array([get_val(run_params, 'AC2_Ini_Pos_x'),
                            get_val(run_params, 'AC2_Ini_Pos_y'),
                            get_val(run_params, 'AC2_Ini_Pos_z')])
    AC2_Ini_Vel = np.array([get_val(run_params, 'AC2_Ini_Vel_x'),
                            get_val(run_params, 'AC2_Ini_Vel_y'),
                            get_val(run_params, 'AC2_Ini_Vel_z')])

    AC2_GPS_horizontal_std = get_val(run_params, 'AC2_GPS_horizontal_std')
    AC2_GPS_horizontal_auto = get_val(run_params, 'AC2_GPS_horizontal_auto')
    AC2_GPS_vert_std = get_val(run_params, 'AC2_GPS_vert_std')
    AC2_GPS_vert_auto = get_val(run_params, 'AC2_GPS_vert_auto')

    AC2_Ini_Pos_Err = np.array([get_val(run_params, 'AC2_Ini_Pos_Err_x'),
                                get_val(run_params, 'AC2_Ini_Pos_Err_y'),
                                get_val(run_params, 'AC2_Ini_Pos_Err_z')])

    AC2_Ini_Vel_Err = np.array([get_val(run_params, 'AC2_Ini_Vel_Err_x'),
                                get_val(run_params, 'AC2_Ini_Vel_Err_y'),
                                get_val(run_params, 'AC2_Ini_Vel_Err_z')])

    AC2_Ini_RPY = np.array([get_val(run_params, 'AC2_Ini_RPY_r'),
                            get_val(run_params, 'AC2_Ini_RPY_p'),
                            get_val(run_params, 'AC2_Ini_RPY_y')])

    AC2_PhysicsUpdateRate = get_val(run_params, 'AC2_PhysicsUpdateRate')
    AC2_Controller_UpdateRate = get_val(run_params, 'AC2_Controller_UpdateRate')
    AC2_Start_Time = get_val(run_params, 'AC2_Start_Time')

    AC2_PosUpdateRate = get_val(run_params, 'AC2_PosUpdateRate')
    AC2_VelUpdateRate = get_val(run_params, 'AC2_VelUpdateRate')
    AC2_AccelUpdateRate = get_val(run_params, 'AC2_AccelUpdateRate')
    AC2_AirSpdSensorUpdateRate = get_val(run_params, 'AC2_AirSpdSensorUpdateRate')

    # Initialize AC2 Tracking System

    AC2_TrackingUnit_UpdateRate = get_val(run_params, 'AC2_TrackingUnit_UpdateRate')
    AC2_TrackingUnit_Pos_Quant = int(get_val(run_params, 'AC2_TrackingUnit_Pos_Quant'))
    AC2_TrackingUnit_Vel_Quant = int(get_val(run_params, 'AC2_TrackingUnit_Vel_Quant'))
    AC2_TrackingUnit_Avail = get_val(run_params, 'AC2_TrackingUnit_Avail')
    AC2_Latency = get_val(run_params, 'AC2_Latency')

    # Init All Agents/Classes

    # Global Agents Init (Clock, Wind, etc.)

    Wind = WindField(WindUpdateRate, 0,
                     auto_x=Wind_auto_x, auto_y=Wind_auto_y, auto_z=Wind_auto_z,
                     std_x=Wind_std_x, std_y=Wind_std_y, std_z=Wind_std_z,
                     mean_x=Wind_mean_x, mean_y=Wind_mean_y, mean_z=Wind_mean_z)

    clock = GlobalClock(update_rate=GlobalPhysicsUpdateRate, stop=GlobalSimStopTime, start=0)

    # AC1 Flight Agents Init

    AC1_AircraftType = AircraftType(mass=AC1_Mass, prop_diameter=AC1_Prop_Diameter, max_cruise_spd=20)
    AC1_DragModel = DragModel(AC1_AircraftType, disabled=False)

    AC1_State = State(AC1_PhysicsUpdateRate, AC1_Start_Time,
                      AC1_Ini_Pos, AC1_Ini_Vel,
                      AC1_Ini_Pos_Err, AC1_Ini_Vel_Err,
                      Wind.get_windspd(AC1_Start_Time, AC1_Ini_Pos), AC1_AircraftType,
                      rpy=AC1_Ini_RPY)

    AC1_FlightPlan, AC1_WPTs = FP_gen(AC1_HDG, AC1_Cruise_Speed, Cruise_Leg_Time, Num_Legs + 1, AC1_Ini_Pos)

    AC1_Controller = MultiRotorController(AC1_Controller_UpdateRate, AC1_Start_Time, AC1_FlightPlan, AC1_AircraftType,
                                          wpt_tol_hor=0, wpt_tol_vert=0, wpt_tol=10)

    AC1_Pos_Nav_Agent = GPSPosNavUpdate(AC1_PosUpdateRate, AC1_Start_Time,
                                        AC1_GPS_horizontal_auto, AC1_GPS_horizontal_auto, AC1_GPS_vert_auto,
                                        AC1_GPS_horizontal_std, AC1_GPS_horizontal_std, AC1_GPS_vert_std,
                                        x_mean=0, y_mean=0, z_mean=0)
    # phase_delay=10 <-- manually set GPS phase delay during start-up (will rand if None)
    AC1_Vel_Nav_Agent = NavUpdate(AC1_VelUpdateRate, AC1_Start_Time)
    AC1_Accel_Nav_Agent = NavUpdate(AC1_AccelUpdateRate, AC1_Start_Time)
    AC1_AirSpeedSensor = NavUpdate(AC1_AirSpdSensorUpdateRate, AC1_Start_Time)

    # AC1 Tracking System Init

    AC1_TrackingUnit = TrackingUnit('AC1', AC1_TrackingUnit_UpdateRate, AC1_Start_Time,
                                    AC1_Latency, latency_distribution=None,
                                    availability=AC1_TrackingUnit_Avail,
                                    pos_quant=AC1_TrackingUnit_Pos_Quant, vel_quant=AC1_TrackingUnit_Vel_Quant,
                                    phase_delay=None)

    # AC2 Flight Agents Init

    AC2_AircraftType = AircraftType(mass=AC2_Mass, prop_diameter=AC2_Prop_Diameter, max_cruise_spd=20)
    AC2_DragModel = DragModel(AC2_AircraftType, disabled=False)

    AC2_State = State(AC2_PhysicsUpdateRate, AC2_Start_Time,
                      AC2_Ini_Pos, AC2_Ini_Vel,
                      AC2_Ini_Pos_Err, AC2_Ini_Vel_Err,
                      Wind.get_windspd(AC2_Start_Time, AC2_Ini_Pos), AC2_AircraftType,
                      rpy=AC2_Ini_RPY)

    AC2_FlightPlan, AC2_WPTs = FP_gen(AC2_HDG, AC2_Cruise_Speed, Cruise_Leg_Time, Num_Legs + 1, AC2_Ini_Pos)

    AC2_Controller = MultiRotorController(AC2_Controller_UpdateRate, AC2_Start_Time, AC2_FlightPlan, AC2_AircraftType,
                                          wpt_tol_hor=0, wpt_tol_vert=0, wpt_tol=10)

    AC2_Pos_Nav_Agent = GPSPosNavUpdate(AC2_PosUpdateRate, AC2_Start_Time,
                                        AC2_GPS_horizontal_auto, AC2_GPS_horizontal_auto, AC2_GPS_vert_auto,
                                        AC2_GPS_horizontal_std, AC2_GPS_horizontal_std, AC2_GPS_vert_std,
                                        x_mean=0, y_mean=0, z_mean=0)
    # phase_delay=10 <-- manually set GPS phase delay during start-up (will rand if None)
    AC2_Vel_Nav_Agent = NavUpdate(AC2_VelUpdateRate, AC2_Start_Time)
    AC2_Accel_Nav_Agent = NavUpdate(AC2_AccelUpdateRate, AC2_Start_Time)
    AC2_AirSpeedSensor = NavUpdate(AC2_AirSpdSensorUpdateRate, AC2_Start_Time)

    # AC2 Tracking System Init

    AC2_TrackingUnit = TrackingUnit('AC2', AC2_TrackingUnit_UpdateRate, AC2_Start_Time,
                                    AC2_Latency, latency_distribution=None,
                                    availability=AC2_TrackingUnit_Avail,
                                    pos_quant=AC2_TrackingUnit_Pos_Quant, vel_quant=AC2_TrackingUnit_Vel_Quant,
                                    phase_delay=None)

    # Centralized Tracking System (Receiver) Init

    Ground_UpdateRate = GlobalPhysicsUpdateRate
    Trackinator = GroundStation(Ground_UpdateRate, 0, phase_delay=0)

    Trackinator.add_airborne_tracker('AC1', AC1_TrackingUnit)
    Trackinator.add_airborne_tracker('AC2', AC2_TrackingUnit)

    # Init Conflict Detection Algorithms

    GT_ConDet = ConflictDetector(GlobalPhysicsUpdateRate, 0, phase_delay=0)

    # Example format for adding distance-based criteria:
    # GT_ConDet.add_conflict_definition('TAG', GT_ConDet.gen_distance_condition(hor_dist_min, vert_dist_min))
    GT_ConDet.add_conflict_definition('NC1', GT_ConDet.gen_distance_condition(15.24, 4.572))
    GT_ConDet.add_conflict_definition('NC2', GT_ConDet.gen_distance_condition(1, 0.5))

    # Example format for adding tau-based criteria:
    # GT_ConDet.add_conflict_definition('TAG', GT_ConDet.gen_tau_mod_condition(tau_mod_min,
    #                                                                          tau_vert_min,
    #                                                                          vert_dist_min,
    #                                                                          DMOD))

    for i in range(10):
        # Add WCV distance-based criteria
        GT_ConDet.add_conflict_definition('WC1' + chr(97 + i),
                                          GT_ConDet.gen_distance_condition(5 * (i + 1), 2.5 * (i + 1)))
        # Add WCV tau-based criteria
        GT_ConDet.add_conflict_definition('WC2' + chr(97 + i),
                                          GT_ConDet.gen_tau_mod_condition(25, 25, 5 * (i + 1), 2.5 * (i + 1)))

    for i in range(5):
        # Add WCV tau-based criteria
        tau_crit = 20 + 5 * i
        GT_ConDet.add_conflict_definition('WC3' + chr(97 + i),
                                          GT_ConDet.gen_tau_mod_condition(tau_crit, tau_crit, 20, 10))

    # Let's try to simulate this...

    while clock.time <= clock.stop:
        clock.update()
        if AC1_State.simstate + AC2_State.simstate == 0:
            continue
        # print(clock.time)
        AC1_State.update(clock.time, AC1_DragModel,
                         AC1_Controller,
                         AC1_Pos_Nav_Agent, AC1_Vel_Nav_Agent, AC1_Accel_Nav_Agent,
                         AC1_AirSpeedSensor,
                         Wind)
        AC2_State.update(clock.time, AC2_DragModel,
                         AC2_Controller,
                         AC2_Pos_Nav_Agent, AC2_Vel_Nav_Agent, AC2_Accel_Nav_Agent,
                         AC2_AirSpeedSensor,
                         Wind)
        AC1_TrackingUnit.update_and_queue(clock.time, AC1_State,
                                          pos_error_agent=None, vel_error_agent=None)
        AC2_TrackingUnit.update_and_queue(clock.time, AC2_State,
                                          pos_error_agent=None, vel_error_agent=None)
        Trackinator.track_and_update_trajectories(clock.time)
        # Check if there is conflict based on ground truth position/velocity
        GT_ConDet.update_r_hor_sq_vert_dist(AC1_State.gt_pos, AC2_State.gt_pos)
        GT_ConDet.update_taus(AC1_State.gt_pos, AC2_State.gt_pos, AC1_State.gt_vel, AC2_State.gt_vel)
        GT_ConDet.conflict_check(clock.time)

    # After running the simulation, get trajectory info
    AC1_GT_Trajectory = AC1_State.get_trajectory()
    AC2_GT_Trajectory = AC2_State.get_trajectory()

    if debug:
        # Return trajectories and wpts for debugging purposes
        AC1_Trk_Trajectory = special_concat(Trackinator.trajectories['AC1'].trajectory)
        AC2_Trk_Trajectory = special_concat(Trackinator.trajectories['AC2'].trajectory)
        return (AC1_GT_Trajectory, AC2_GT_Trajectory, AC1_Trk_Trajectory, AC2_Trk_Trajectory, AC1_WPTs, AC2_WPTs)

    # Do a quick check if final AC positions are roughly near their target WPTs
    AC1_Dist_to_Last_WPT = np.linalg.norm(AC1_State.gt_pos - AC1_WPTs[-1])
    AC2_Dist_to_Last_WPT = np.linalg.norm(AC2_State.gt_pos - AC2_WPTs[-1])

    res = pd.DataFrame({'Run': [run],
                        'AC1_NSE_Radial_Error_Mean': [AC1_GT_Trajectory['gt_hor_err'].mean()],
                        'AC1_NSE_Radial_Error_Std': [AC1_GT_Trajectory['gt_hor_err'].std()],
                        'AC1_NSE_Mean_x': [extract_from_df(AC1_GT_Trajectory, 'gt_pos_err', 0).mean()],
                        'AC1_NSE_Mean_y': [extract_from_df(AC1_GT_Trajectory, 'gt_pos_err', 1).mean()],
                        'AC1_NSE_Mean_z': [extract_from_df(AC1_GT_Trajectory, 'gt_pos_err', 2).mean()],
                        'AC1_NSE_Std_x': [extract_from_df(AC1_GT_Trajectory, 'gt_pos_err', 0).std()],
                        'AC1_NSE_Std_y': [extract_from_df(AC1_GT_Trajectory, 'gt_pos_err', 1).std()],
                        'AC1_NSE_Std_z': [extract_from_df(AC1_GT_Trajectory, 'gt_pos_err', 2).std()],

                        'AC2_NSE_Radial_Error_Mean': [AC2_GT_Trajectory['gt_hor_err'].mean()],
                        'AC2_NSE_Radial_Error_Std': [AC2_GT_Trajectory['gt_hor_err'].std()],
                        'AC2_NSE_Mean_x': [extract_from_df(AC2_GT_Trajectory, 'gt_pos_err', 0).mean()],
                        'AC2_NSE_Mean_y': [extract_from_df(AC2_GT_Trajectory, 'gt_pos_err', 1).mean()],
                        'AC2_NSE_Mean_z': [extract_from_df(AC2_GT_Trajectory, 'gt_pos_err', 2).mean()],
                        'AC2_NSE_Std_x': [extract_from_df(AC2_GT_Trajectory, 'gt_pos_err', 0).std()],
                        'AC2_NSE_Std_y': [extract_from_df(AC2_GT_Trajectory, 'gt_pos_err', 1).std()],
                        'AC2_NSE_Std_z': [extract_from_df(AC2_GT_Trajectory, 'gt_pos_err', 2).std()],

                        'AC1_Dist_to_Last_WPT': [AC1_Dist_to_Last_WPT],
                        'AC2_Dist_to_Last_WPT': [AC2_Dist_to_Last_WPT],
                        'AC1_EndState_Time': [AC1_State.time],
                        'AC2_EndState_Time': [AC2_State.time],
                        'Total_Flight_Time': [AC1_State.time + AC2_State.time]
                        })
    for tag in GT_ConDet.conflict_definitions.keys():
        res[tag + '_Start_Time'] = [GT_ConDet.get_earliest_conflict_start_time(tag)]
        res[tag + '_End_Time'] = [GT_ConDet.get_latest_conflict_end_time(tag)]
        res[tag + '_Detected'] = [GT_ConDet.history_of_conflict(tag) & 1]

    return res


