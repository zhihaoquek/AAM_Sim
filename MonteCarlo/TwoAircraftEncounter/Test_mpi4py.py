# -*- coding: utf-8 -*-
"""
@Time    : 03/7/2022 9:41 AM
@Author  : Zhi Hao
@FileName: Test_Run.py
@Description: Script for reading input parameters and running MultiProcessing Monte Carlo Simulation.
@Package dependency:
"""

import numpy as np
import os
import sys
cur_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(cur_dir)
sys.path.append(os.path.dirname(os.path.dirname(cur_dir)))  # Add path to directory
sys.path.append(os.getcwd())  # Add directory to path
import pandas as pd
from CrossPlatformDev import my_print, join_str
from ScenarioMP_Sensitivity_Analysis_v2 import simulate_encounter
from mpi4py.futures import MPIPoolExecutor
# from mpi4py import MPI
import psutil
# from tqdm import tqdm
import time
os.chdir('..')
os.chdir('..')

import warnings
warnings.filterwarnings('ignore') # <---- hides warnings, makes tqdm work better.

Init_Param_Path = join_str(os.getcwd(), 'MonteCarlo', 'TwoAircraftEncounter', 'Init_Param_Sensitivity_Analysis.csv')
data = pd.read_csv(Init_Param_Path)

data = data['Run'].unique()[0:12]
# print('Number of available CPU cores: %s'%psutil.cpu_count(logical=True))
# comm = MPI.COMM_WORLD
# universe_size=comm.Get_attr(MPI.UNIVERSE_SIZE)
# print("universe size is ", universe_size)
if __name__ == '__main__':
    start = time.time()
    executor = MPIPoolExecutor(max_workers=8)
    results = executor.map(simulate_encounter, data)
    results = pd.concat(results)
    end = time.time()
    compute_time = end - start
    simulated_flight_time_s = results['Total_Flight_Time'].sum()
    speedup = simulated_flight_time_s / compute_time
    print('Took %.3f seconds, simulated flight time is %.3f seconds, total speedup is %.3f times' % (end - start,
                                                                                                     simulated_flight_time_s,
                                                                                                     speedup))

    Results_Path = join_str(os.getcwd(), 'MonteCarlo', 'TwoAircraftEncounter',
                            'Results_Tracking_Sensitivity_Analysis_v2.csv')

    results.to_csv(Results_Path)