class AntiWindupPIDController:
    """PID with standard industrial anti-windup. Drop-in replacement for
    PIDController (same attribute/method surface: kp/ki/kd, setpoint,
    output_limits, _integral, _prev_error, reset(), update()).

    Modes:
      - "backcalc": back-calculation (Astrom & Hagglund 1995). When the raw
        output saturates, the integral state is driven back toward the value
        consistent with the saturated output:
            integral += e*dt + (dt / (Tt * ki)) * (u_sat - u_raw)
        Tt is the tracking time constant (s).
      - "clamp": conditional integration. Integration is skipped when the
        output is saturated AND the error would push it further into
        saturation.
      - "none": behaves exactly like the plain PIDController.
    """

    def __init__(self, kp=1.0, ki=0.0, kd=0.0, setpoint=0.0, output_limits=(None, None), mode="backcalc", tt=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self.mode = str(mode)
        self.tt = float(tt)

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
        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        d_term = self.kd * derivative

        # Raw (unsaturated) output using the integral state as of last step
        # plus this step's tentative integration.
        tentative_integral = self._integral + error * dt
        i_term = self.ki * tentative_integral
        u_raw = p_term + i_term + d_term

        min_limit, max_limit = self.output_limits
        u_sat = u_raw
        if min_limit is not None:
            u_sat = max(min_limit, u_sat)
        if max_limit is not None:
            u_sat = min(max_limit, u_sat)
        saturated = u_sat != u_raw

        if self.mode == "clamp":
            # Conditional integration: only accept this step's integration if
            # not saturated, or if the error drives the output OUT of saturation.
            if not saturated or (error * (u_sat - u_raw)) > 0.0:
                self._integral = tentative_integral
            # else: keep previous integral (skip integration this step)
        elif self.mode == "backcalc":
            self._integral = tentative_integral
            if saturated and self.ki > 1e-9:
                # Drive integral toward consistency with the saturated output.
                self._integral += (dt / (self.tt * self.ki)) * (u_sat - u_raw)
        else:  # "none"
            self._integral = tentative_integral

        # Recompute the reported terms from the (possibly corrected) state.
        i_term = self.ki * self._integral
        output = p_term + i_term + d_term
        if min_limit is not None:
            output = max(min_limit, output)
        if max_limit is not None:
            output = min(max_limit, output)

        self._prev_error = error
        return output, (p_term, i_term, d_term)


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
        """
        Update the PID controller.
        Allows dynamic updating of gains (Meta-RL use case).
        """
        if kp is not None:
            self.kp = kp
        if ki is not None:
            self.ki = ki
        if kd is not None:
            self.kd = kd

        error = self.setpoint - measurement

        # Proportional term
        p_term = self.kp * error

        # Integral term
        self._integral += error * dt
        i_term = self.ki * self._integral

        # Derivative term
        # Simple finite difference; could be improved with filtering
        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        d_term = self.kd * derivative

        # Total output
        output = p_term + i_term + d_term

        # Clamp output
        min_limit, max_limit = self.output_limits
        if min_limit is not None:
            output = max(min_limit, output)
        if max_limit is not None:
            output = min(max_limit, output)

        self._prev_error = error

        return output, (p_term, i_term, d_term)
