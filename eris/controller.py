""" This module implements latency-based resource controllers """

from mresource import Resource
from collections import deque
from datetime import datetime

CONTROLLER_MAP = {
    'proportional': ProportionalController,
    'step': StepController,
}


class Controller(object):
    """ Generic latency-based controller """

    BUFFER_SIZE = 1024

    def __init__(self, cpuq, llc, lat_thresh, margin_ratio):
        self.cpuq = cpuq
        self.llc = llc
        self.lat_thresh = lat_thresh
        self.margin_ratio = margin_ratio
        self.buffer = deque(maxlen=self.BUFFER_SIZE)

    def update(self, be_containers, lc_containers, lat):
        """
        Update the Controller with the currently active BE and LC containers,
        and the current latency of the LC service. Decide how the CPU quota and
        LLC resources should be partitioned.
        """
        pass


def step_up(resource, be_containers, lc_containers):
    if not resource.is_full_level():
        resource.increase_level()
        resource.budgeting(be_containers, lc_containers)
        print(f"{datetime.now().isoformat(' ')} Increasing BE jobs to level {resource.quota_level}")

def step_down(resource, be_containers, lc_containers):
    if not resource.is_min_level():
        resource.reduce_level(level)
        resource.budgeting(be_containers, lc_containers)
        print(f"{datetime.now().isoformat(' ')} Decreasing BE jobs to level {resource.quota_level}")

def step_by(resource, be_containers, lc_containers, delta):
    if delta == 0:
        return
    if resource.is_full_level():
        if delta < 0:
            # Treat FULL as MAX + 1
            # level = resource.level_max + 1 + delta
            # Treat FULL as = MAX (this is what the default resource code does?)
            level = resource.level_max + delta
        else: # Already FULL, do nothing
            return
    else:
        level = max(resource.BUDGET_LEV_MIN, resource.quota_level + delta)
        if level >= resource.level_max: # >= max = full
            level = resource.BUDGET_LEV_FULL
    
    resource.set_level(level)
    resource.budgeting(be_containers, lc_containers)
    print(f"{datetime.now().isoformat(' ')} Setting BE jobs to level {resource.quota_level}")


class ProportionalController(Controller):
    """ Linear scaling controller """
    CYCLE_THRESHOLD = 3

    def __init__(self, cpuq, llc, lat_thresh, margin_ratio):
        super().__init__(cpuq, llc, lat_thresh, margin_ratio)
        self.cycles = 0

    def _level_estimate(self, lat):
        latency_diff = lat - self.lat_thresh
        level_change = floor(-8 * latency_diff / self.lat_thresh)
        print(f"{datetime.now().isoformat(' ')} Latency: {lat}, LatDiff: {latency_diff}, Levels: {level_change}")
        return level_change

    def update(self, be_containers, lc_containers, lat):
        level_change = self._level_estimate(lat)

        if level_change < 0:
            self.cycles = 0
            step_by(self.cpuq, be_containers, lc_containers, level_change)
        
        if level_change > 0:
            self.cycles += 1
            if self.cycles >= self.CYCLE_THRESHOLD:
                self.cycles = 0
                step_by(self.cpuq, be_containers, lc_containers, level_change)


class StepController(Controller):
    """ Single-step controller """
    CYCLE_THRESHOLD = 3

    def __init__(self, cpuq, llc, lat_thresh, margin_ratio):
        super().__init__(cpuq, llc, lat_thresh, margin_ratio)
        self.cycles = 0

    def _level_estimate(self, lat):
        latency_diff = lat - self.lat_thresh
        level_change = floor(-8 * latency_diff / self.lat_thresh)
        print(f"{datetime.now().isoformat(' ')} Latency: {lat}, LatDiff: {latency_diff}, Levels: {level_change}")
        return level_change

    def update(self, be_containers, lc_containers, lat):
        level_change = self._level_estimate(lat)

        if level_change < 0:
            self.cycles = 0
            step_down(self.cpuq, be_containers, lc_containers)

        if level_change > 0:
            self.cycles += 1
            if self.cycles >= self.CYCLE_THRESHOLD:
                self.cycles = 0
                step_up(self.cpuq, be_containers, lc_containers)


# def detect_margin_exceed(self, latency, lc_utils, be_utils):
#     """
#     Detect if BE workload utilization exceed the safe margin
#         lc_utils - utilization of all LC workloads
#         be_utils - utilization of all BE workloads
#     """
#     margin = latency * self.min_margin_ratio
#     beq = self.cpu_quota

#     if self.verbose:
#         print(datetime.now().isoformat(' ') + ' lcUtils: ', lc_utils,
#               ' beUtils: ', be_utils, ' beq: ', beq, ' margin: ', margin)

    
#     exceed = latency > self.lat_threshold
#     hold = margin > self.lat_threshold and not exceed

#     if self.verbose:
#         if exceed:
#             print(f"{datetime.now().isoformat(' ')} exceeding threshold {self.lat_threshold}")
#         if hold:
#             print(f"{datetime.now().isoformat(' ')} holding at latency {margin}")

#     return (exceed, hold)

