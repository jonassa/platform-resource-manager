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

""" This module implements CPU cycle control based on CFS quota """

from __future__ import print_function
from __future__ import division

import subprocess
from datetime import datetime
from mresource import Resource



class CpuQuota(Resource):
    """ This class is the resource class of CPU cycle """
    CPU_QUOTA_DEFAULT = -1
    CPU_QUOTA_MIN = 1000
    CPU_QUOTA_CORE = 100000
    CPU_QUOTA_PERCENT = CPU_QUOTA_CORE / 100
    CPU_QUOTA_HALF_CORE = CPU_QUOTA_CORE * 0.5
    CPU_SHARE_BE = 2
    CPU_SHARE_LC = 200000
    PREFIX = '/sys/fs/cgroup/cpu/'

    def __init__(self, sysMaxUtil, minMarginRatio, verbose, lat_threshold):
        super(CpuQuota, self).__init__(lat_threshold)
        self.min_margin_ratio = minMarginRatio
        self.update_max_sys_util(sysMaxUtil)
        self.lat_threshold = lat_threshold
        self.update()
        self.verbose = verbose

    def update(self):
        if self.is_full_level():
            self.cpu_quota = CpuQuota.CPU_QUOTA_DEFAULT
        elif self.is_min_level():
            self.cpu_quota = CpuQuota.CPU_QUOTA_MIN
        else:
            self.cpu_quota = self.quota_level * int(self.quota_step)

    def update_max_sys_util(self, lc_max_util):
        """
        Update quota max and step based on given LC system maximal utilization
        monitored
            lc_max_util - maximal LC workloads utilization monitored
        """
        self.quota_max = 900 * CpuQuota.CPU_QUOTA_PERCENT
        self.quota_step = self.quota_max / Resource.BUGET_LEV_MAX

    @staticmethod
    def __get_cfs_period(container):
        path = CpuQuota.PREFIX + container.parent_path + container.con_path +\
                '/cpu.cfs_period_us'
        with open(path) as perdf:
            res = perdf.readline()
        try:
            period = int(res)
            return period
        except ValueError:
            return 0

    def __set_quota(self, container, quota):
        period = self.__get_cfs_period(container)
        if period != 0 and quota != CpuQuota.CPU_QUOTA_DEFAULT\
           and quota != CpuQuota.CPU_QUOTA_MIN:
            rquota = int(quota * period / CpuQuota.CPU_QUOTA_CORE)
        else:
            rquota = quota

        path = CpuQuota.PREFIX + container.parent_path +\
            container.con_path + '/cpu.cfs_quota_us'
        with open(path, 'w') as shrf:
            shrf.write(str(rquota))
        print(datetime.now().isoformat(' ') + ' set container ' +
              container.name + ' cpu quota to ' + str(rquota))

    @staticmethod
    def set_share(container, share):
        """
        Set CPU share in container
            share - given CPU share value
        """
        path = CpuQuota.PREFIX + container.parent_path +\
            container.con_path + '/cpu.shares'
        with open(path, 'w') as shrf:
            shrf.write(str(share))

        print(datetime.now().isoformat(' ') + ' set container ' +
              container.name + ' cpu share to ' + str(share))

    def budgeting(self, bes, lcs):
        newq = int(self.cpu_quota / len(bes))
        for con in bes:
            if self.is_min_level() or self.is_full_level():
                self.__set_quota(con, self.cpu_quota)
            else:
                self.__set_quota(con, newq)

    def detect_margin_exceed(self, latency, lc_utils, be_utils):
        """
        Detect if BE workload utilization exceed the safe margin
            lc_utils - utilization of all LC workloads
            be_utils - utilization of all BE workloads
        """
        margin = latency * self.min_margin_ratio
        beq = self.cpu_quota

        if self.verbose:
            print(datetime.now().isoformat(' ') + ' lcUtils: ', lc_utils,
                  ' beUtils: ', be_utils, ' beq: ', beq, ' margin: ', margin)

        
        exceed = latency > self.lat_threshold
        hold = margin > self.lat_threshold and not exceed

        if self.verbose:
            if exceed:
                print(f"{datetime.now().isoformat(' ')} exceeding threshold {self.lat_threshold}")
            if hold:
                print(f"{datetime.now().isoformat(' ')} holding at latency {margin}")

        return (exceed, hold)
