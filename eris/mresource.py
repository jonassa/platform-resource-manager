# Copyright (C) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
#
#
# SPDX-License-Identifier: Apache-2.0
""" This module defines general resource control methods """

from math import floor, ceil
from datetime import datetime


class Resource(object):
    """ Resource Class is abstraction of resource """
    BUDGET_LEV_FULL = -1
    BUDGET_LEV_MIN = 0
    BUDGET_LEV_MAX = 20

    def __init__(self, init_level=BUDGET_LEV_MIN, level_max=BUDGET_LEV_MAX):
        self.quota_level = init_level
        self.level_max = level_max # this is necessary because LLC sets max level based on cbm

    def is_min_level(self):
        """ is resource controled in lowest level """
        return self.quota_level == Resource.BUDGET_LEV_MIN

    def is_full_level(self):
        """ is resource controled in full level """
        return self.quota_level == Resource.BUDGET_LEV_FULL

    def set_level(self, level):
        """ set resource in given level """
        self.quota_level = level

    def increase_level(self):
        """ increase resource level by one step"""
        self.quota_level += 1
        if self.quota_level >= self.level_max:
            self.quota_level = Resource.BUDGET_LEV_FULL
    
    def reduce_level(self):
        """ reduce resource level by one step """
        if self.is_full_level():
            self.quota_level = self.level_max
        elif self.is_min_level():
            pass
        else:
            self.quota_level -= 1

    def budgeting(self, bes, lcs):
        """ set real resource levels based on quota level """
        pass
