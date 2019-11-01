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


class WorkloadFilter(object):
    """
    Support user defined filter criteria for model building
    Arguments:
        cpu_quota: assigned cpu quota filter, workloads with assigned cpu quota less than
                    this filter will be ignored
        name_pattern: regular express pattern to filter workloads with name, workloads name
                    does not follow the pattern will be ignored
    """
    def __init__(
        self,
        cpu_quota: float = 0.0,
        name_pattern: str = None
    ):
        self.cpu_quota = cpu_quota
        self.name_pattern = name_pattern
