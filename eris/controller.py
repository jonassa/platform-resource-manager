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


class AdaptiveController(Controller):

    # Cycles to monitor / measurement sample size
    BUFFER_SIZE = 10

    SLACK_HIGH = 0.50
    SLACK_MID = 0.30
    SLACK_LOW = 0.15
    SLACK_HOLD = 0.10
    SLACK_MARGIN = 0.05

    def __init__(self, cpuq, llc, target, margin):
        super().__init__(cpuq, llc, target, margin)
        self.cycles = 0

    def update(self, be_containers, lc_containers, lat):
        self.be_containers = be_containers
        self.lc_containers = lc_containers

        slack = (self.target - lat) / self.target
        self.buffer.append(slack)
        self.regulate(slack)

    def _status(self, msg):
        print(msg, ':', f"slack={self.buffer[-1]}, level={self.cpuq.quota_level}, cycles={self.cycles}")

    def _measure(self):
        return sum(self.buffer) / len(self.buffer)

    def regulate(self, slack):
        if slack < self.SLACK_MARGIN:
            self._status("VIOLATION")
            self.cycles = 0
            self.step_to(self.cpuq, self.cpuq.BUDGET_LEV_MIN)
        else:
            self.cycles += 1
            if self.cycles >= self.BUFFER_SIZE:
                self.cycles = 0
                mean = self._measure()
                print(f"Mean slack: {mean} at level {self.cpuq.quota_level}")
                if mean < self.SLACK_HOLD:
                    self._status("HOLD")
                elif mean < self.SLACK_LOW:
                    self._status("Step low")
                    self.step_up(self.cpuq)
                elif mean < self.SLACK_MID:
                    self._status("Step mid")
                    self.step_by(self.cpuq, 5)
                elif mean < self.SLACK_HIGH:
                    self._status("Step high")
                    self.step_by(self.cpuq, 10)
                else:
                    self._status("Step jump")
                    self.step_by(self.cpuq, 25)


class BinaryController(Controller):
    """ Gets stuck, could fix """

    LOWER_SLACK  = 0.1
    WAIT_CYCLES  = 5

    def __init__(self, cpuq, llc, target, margin):
        super().__init__(cpuq, llc, target, margin)
        self.cycles = 0
        self.mark = cpuq.level_max
        self.pre_level = cpuq.quota_level
        self.recovery = False

    def update(self, be_containers, lc_containers, lat):
        self.be_containers = be_containers
        self.lc_containers = lc_containers

        slack = (self.target - lat) / self.target
        self.buffer.append(slack)
        self.regulate(slack)

    def regulate(self, slack):
        if slack < self.LOWER_SLACK:
            self.cycles = 0
            self.mark = self.cpuq.quota_level
            if self.recovery:
                self.step_to(self.cpuq, self.cpuq.BUDGET_LEV_MIN)
            else:
                self.step_to(self.cpuq, self.pre_level)
                self.recovery = True
            print(f"Violation at level {self.mark} with {slack} slack. Stepping back to pre_level={self.pre_level}")
        else:
            self.cycles += 1
            if self.cycles >= self.WAIT_CYCLES:
                self.cycles = 0
                self.recovery = False
                range = self.mark - self.cpuq.quota_level
                if (range <= 0):
                    print(f"Ignoring RANGE={range}!")
                else:
                    print(f"Stepping up by half-range (range={range})")
                    self.pre_level = self.cpuq.quota_level
                    self.step_to(self.cpuq, self.cpuq.quota_level + int(range/2))
            else:
                print(f"No violation, incrementing cycles to {self.cycles}")




class PIDController(Controller):
    """ PID-like controller """

    # TODO: either do proper tuning, or come up with some way to estimate while running
    Kp, Ki, Kd = (1.0, 1.0, 1.0)

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

