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
import multiprocessing as mp
import psutil
from tqdm import tqdm
import time
os.chdir('..')
os.chdir('..')

import warnings
warnings.filterwarnings('ignore') # <---- hides warnings, makes tqdm work better.

Init_Param_Path = join_str(os.getcwd(), 'MonteCarlo', 'TwoAircraftEncounter', 'Init_Param_Sensitivity_Analysis.csv')
data = pd.read_csv(Init_Param_Path)

data = data['Run'].unique()[0:8]
print('Number of available CPU cores: %s'%psutil.cpu_count(logical=True))
if __name__ == '__main__':
    start = time.time()
    # with mp.Pool(processes=2) as pool:
    with mp.Pool(processes=psutil.cpu_count(logical=True)) as pool:
        results = list(tqdm(pool.imap(simulate_encounter, data), total=len(data)))
        pool.close()
        pool.join()
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