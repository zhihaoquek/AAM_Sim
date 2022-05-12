import os

print(os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# from ScenarioMP_Sensitivity_Analysis_v2 import simulate_encounter
os.chdir('..')
os.chdir('..')
print(os.getcwd())
print(os.listdir())
import sys
sys.path.append(os.getcwd())
from CrossPlatformDev import my_print, join_str

