""" This module implements latency-based resource controllers """

from collections import deque
from datetime import datetime
from math import floor, ceil


class Controller():
    """ Generic latency-based controller """

    # 4 min, 16 sec with latency interval = 1 sec
    BUFFER_SIZE = 256

    def __init__(self, cpuq, llc, target, margin):
        self.cpuq = cpuq
        self.llc = llc
        self.target = target
        self.margin = margin

        self.buffer = deque(maxlen=self.BUFFER_SIZE)
        self.be_containers = None
        self.lc_containers = None

    def update(self, be_containers, lc_containers, lat):
        self.be_containers = be_containers
        self.lc_containers = lc_containers
        self.buffer.append(lat)
        self.regulate(lat)

    def regulate(self, lat):
        """ Decide how to allocate CPU/LLC based on the current/previous latency, or other metrics """
        pass

    def step_up(self, resource):
        if not resource.is_full_level():
            resource.increase_level()
            resource.budgeting(self.be_containers, self.lc_containers)
            print(f"{datetime.now().isoformat(' ')} Increasing BE CPU quota to level {resource.quota_level}")

    def step_down(self, resource):
        if not resource.is_min_level():
            resource.reduce_level()
            resource.budgeting(self.be_containers, self.lc_containers)
            print(f"{datetime.now().isoformat(' ')} Decreasing BE CPU quota to level {resource.quota_level}")

    def step_by(self, resource, delta):
        if delta == 0:
            return
        if resource.is_full_level():
            if delta < 0:
                # Treat FULL as MAX + 1
                level = resource.level_max + 1 + delta
            else:
                return
        else:
            level = max(resource.BUDGET_LEV_MIN, resource.quota_level + delta)
            if level > resource.level_max:
                level = resource.BUDGET_LEV_FULL
        
        resource.set_level(level)
        resource.budgeting(self.be_containers, self.lc_containers)
        print(f"{datetime.now().isoformat(' ')} Setting BE CPU quota to level {resource.quota_level}")

    def step_to(self, resource, level):
        level = max(resource.BUDGET_LEV_MIN, level)
        if level > resource.level_max:
            level = resource.BUDGET_LEV_FULL

        resource.set_level(level)
        resource.budgeting(self.be_containers, self.lc_containers)
        print(f"{datetime.now().isoformat(' ')} Setting BE CPU quota to level {resource.quota_level}")


class HeuristicController(Controller):
    """ Logics and stuff """

    LOWER_SLACK  = 0.1
    MIDDLE_SLACK = 0.5
    UPPER_SLACK  = 0.9

    def __init__(self, cpuq, llc, target, margin):
        super().__init__(cpuq, llc, target, margin)

    def update(self, be_containers, lc_containers, lat):
        self.be_containers = be_containers
        self.lc_containers = lc_containers

        slack = (self.target - lat) / self.target
        self.buffer.append(slack)
        self.regulate(slack)

    def regulate(self, slack):
        pass


class PIDController(Controller):
    """ PID-like controller """

    # TODO: either do proper tuning, or come up with some way to estimate while running
    Kp, Ki, Kd = (1.0, 1.0, 1.0)

    # TODO: pass from ctx.args
    SAMPLING_INTERVAL = 1 

    def __init__(self, cpuq, llc, target, margin):
        super().__init__(cpuq, llc, target, margin)

    def update(self, be_containers, lc_containers, lat):
        self.be_containers = be_containers
        self.lc_containers = lc_containers

        error = self.target - lat
        self.buffer.append(error)
        self.regulate(error)

    def regulate(self, error):
        proportional = self.Kp * error
        integral = self.Ki * sum(self.buffer) * self.SAMPLING_INTERVAL

        try:
            derivative = self.Kd * (error - self.buffer[-2]) / self.SAMPLING_INTERVAL
        except IndexError:
            derivative = 0

        output = proportional + integral + derivative

        # TODO: map output to quota level
        print(f"output={output}")


class ProportionalController(Controller):
    """ Linear scaling controller """

    CYCLE_THRESHOLD = 3

    def __init__(self, cpuq, llc, target, margin):
        super().__init__(cpuq, llc, target, margin)
        self.target = target * margin
        self.cycles = 0

    def _level_estimate(self, lat):
        latency_diff = lat - self.target
        level_change = floor(-8 * latency_diff / self.target)
        print(f"{datetime.now().isoformat(' ')} Latency: {lat}, LatDiff: {latency_diff}, Level change: {level_change}")
        return level_change

    def regulate(self, lat):
        level_change = self._level_estimate(lat)

        if level_change < 0:
            self.cycles = 0
            self.step_by(self.cpuq, level_change)
        
        if level_change > 0:
            self.cycles += 1
            if self.cycles >= self.CYCLE_THRESHOLD:
                self.cycles = 0
                self.step_by(self.cpuq, level_change)


class StepController(Controller):
    """ Single-step controller """

    CYCLE_THRESHOLD = 3

    def __init__(self, cpuq, llc, target, margin):
        super().__init__(cpuq, llc, target, margin)
        self.target = target * margin
        self.cycles = 0

    def _level_estimate(self, lat):
        latency_diff = lat - self.target
        level_change = floor(-8 * latency_diff / self.target)
        print(f"{datetime.now().isoformat(' ')} Latency: {lat}, LatDiff: {latency_diff}, Level change: {level_change}")
        return level_change

    def regulate(self, lat):
        level_change = self._level_estimate(lat)

        if level_change < 0:
            self.cycles = 0
            self.step_down(self.cpuq)

        if level_change > 0:
            self.cycles += 1
            if self.cycles >= self.CYCLE_THRESHOLD:
                self.cycles = 0
                self.step_up(self.cpuq)


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

