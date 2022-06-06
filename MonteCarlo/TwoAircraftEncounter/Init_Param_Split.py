# -*- coding: utf-8 -*-
"""
@Time    : 03/7/2022 9:41 AM
@Author  : Zhi Hao
@FileName: Init_Param_Split.py
@Description: Script to split Init_Param_Sensitivity_Analysis.csv files into small parts for parallel job execution.
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
os.chdir('..')
os.chdir('..')

def split(num_splits):
    if num_splits <=1:
        print('Num_splits must be >= 2')
        return None
    else:
        Init_Param_Path = join_str(os.getcwd(), 'MonteCarlo', 'TwoAircraftEncounter',
                                   'Init_Param_Sensitivity_Analysis.csv')
        data = pd.read_csv(Init_Param_Path)
        num_runs = len(data)
        if num_runs % num_splits == 0:
            runs_per_split = num_runs // num_splits
        else:
            runs_per_split = num_runs // num_splits + 1
        for split_num in range(num_splits-1):
            temp_df = data[(data['Run'] >= (split_num*runs_per_split)) &
                           (data['Run'] <= ((split_num+1)*runs_per_split-1))]
            temp_path = 'Init_Param_Sensitivity_Analysis_' + str(split_num) + '.csv'
            temp_df.to_csv(join_str(os.getcwd(), 'MonteCarlo', 'TwoAircraftEncounter',
                                    temp_path))
            print('Split no.: %.0f, runs: %.0f <= run no. <= %.0f' % (split_num, split_num * runs_per_split,
                                                                      (split_num + 1) * runs_per_split - 1))
        # Handle the 'last' split...
        temp_df = data[(data['Run'] >= (runs_per_split*(num_splits-1))) & (data['Run'] <= num_runs-1)]
        temp_path = 'Init_Param_Sensitivity_Analysis_' + str(num_splits-1) + '.csv'
        temp_df.to_csv(join_str(os.getcwd(), 'MonteCarlo', 'TwoAircraftEncounter',
                                temp_path))
        print('Split no.: %.0f, runs: %.0f <= run no. <= %.0f' % (num_splits - 1,
                                                                  runs_per_split * (num_splits - 1), num_runs - 1))
