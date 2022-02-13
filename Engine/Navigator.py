# -*- coding: utf-8 -*-
"""
@Time    : 02/11/2022 9:41 AM
@Author  : Zhi Hao
@FileName: Navigator.py
@Description: Navigator agent.
@Package dependency:
"""
import numpy as np
import CrossPlatformDev
from Engine.GlobalClock import Agent


class NavUpdate(Agent):
    def __init__(self, update_rate, start_time, err=np.zeros(3)):
        super().__init__(update_rate, start_time)
        self.err = err

    def update_error(self, time):
        if super().check_time(time):
            self.err = np.zeros(3)
        return self.err