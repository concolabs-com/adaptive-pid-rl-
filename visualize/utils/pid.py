import numpy as np


class PIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, setpoint=0.0, output_limits=(None, None)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_limits = output_limits

        self._prev_error = 0.0
        self._integral = 0.0

    def reset(self):
        self._prev_error = 0.0
        self._integral = 0.0

    def update(self, measurement, dt, kp=None, ki=None, kd=None):
        if kp is not None:
            self.kp = kp
        if ki is not None:
            self.ki = ki
        if kd is not None:
            self.kd = kd

        error = self.setpoint - measurement

        p_term = self.kp * error

        self._integral += error * dt
        i_term = self.ki * self._integral

        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        d_term = self.kd * derivative

        output = p_term + i_term + d_term

        min_limit, max_limit = self.output_limits
        if min_limit is not None:
            output = max(min_limit, output)
        if max_limit is not None:
            output = min(max_limit, output)

        self._prev_error = error

        return output, (p_term, i_term, d_term)
