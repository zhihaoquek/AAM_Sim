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