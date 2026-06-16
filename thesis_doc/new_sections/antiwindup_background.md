# New Section (Ch. 2): Integral Windup and Anti-Windup Compensation

> Integration target: new §2.y after the PID subsection. Replaces the plan to
> discuss windup only in Ch. 5 — after audit finding F5 the anti-windup
> baseline is a first-class comparator and needs proper background. Equations
> match `utils/pid.py` (`AntiWindupPIDController`).

## 2.y.1 The windup mechanism

The textbook PID law

$$ u(t) = K_p e(t) + K_i \int_0^t e(\tau)\, d\tau + K_d \dot{e}(t) $$

assumes the commanded $u$ is realised by the actuator. Real actuators
saturate: $u_{\text{sat}} = \mathrm{clip}(u, u_{\min}, u_{\max})$ — here the
motor command is limited to $[-1, 1]$ and the joint torque to
$\pm 0.5\,\kappa$ N·m. During a long approach the error keeps one sign, the
integrator accumulates

$$ I(T) = K_i \int_0^T e\, d\tau \approx K_i\, \bar{e}\, T, $$

and by the time the vehicle reaches the target the integral term alone can
exceed the entire actuator range. Braking requires the derivative term to
dominate: $K_d |\dot{e}| > I$. For a 5 m approach at the actuator-limited
cruise speed (≈ 0.5 m/s, $T \approx 10$ s, $\bar e \approx 2.5$ m), the
integral reaches $I \approx K_i \cdot 25 \approx 17.5$ for $K_i = 0.7$ —
saturating the actuator by itself — while the maximum braking contribution at
$|\dot e| = 0.5$ m/s and $K_d = 0.5$ is just $0.25$. The result is the large
overshoot-and-recovery transient measured in Chapter 5 (≈ 7 m overshoot,
settling 74–93 s where recovery occurs at all).

## 2.y.2 Classical anti-windup

Industrial PID implementations therefore include anti-windup compensation
(Åström and Hägglund, 1995). Two standard mechanisms are used in this thesis:

**Back-calculation.** When the output saturates, the integral state is driven
toward the value consistent with the saturated output through a tracking term
with time constant $T_t$:

$$ \dot{I} = K_i\, e + \frac{1}{T_t} \left( u_{\text{sat}} - u \right). $$

When $u = u_{\text{sat}}$ the law reduces to ordinary integration; in
saturation the second term bleeds the integrator at a rate set by $T_t$. The
common heuristic $T_t \approx \sqrt{T_i T_d}$ gives $T_t \approx 0.85$ s for
the gains used here; $T_t = 1$ s is used throughout.

**Conditional integration (clamping).** Integration is suspended whenever the
output is saturated *and* the error drives it further into saturation:

$$ \dot{I} = \begin{cases} 0 & \text{if } u \neq u_{\text{sat}} \text{ and } e\,(u_{\text{sat}} - u) \le 0, \\ K_i\, e & \text{otherwise.} \end{cases} $$

Both mechanisms are two to five lines of code and ship in every industrial
PID block (PLC function blocks, motor-drive firmware). A comparison of
learning-based control against PID *without* anti-windup therefore
overstates the learning method's advantage; conversely an environment aid
that zeroes the integrator with task knowledge (the `brake_integral_reset`
mechanism, §3.x) understates it. The fair classical baseline sits between
these extremes and is evaluated explicitly in Chapter 5.

## 2.y.3 Relation to the environment's integral-reset aid

The `brake_integral_reset` mechanism used during development zeroes the
integrator once the vehicle enters the braking zone ($|e| < 2$ m). It can be
read as a crude, *task-aware* conditional integration: instead of consulting
actuator saturation it consults distance-to-target — information a generic
PID block does not have. The audit (Ch. 5, finding F5) shows the consequences:
with the aid, plain PID and anti-windup PID are indistinguishable; without
the aid, plain PID suffers catastrophic windup transients while anti-windup
PID retains 100% task success at a ~2× settling-time cost relative to the
aided configuration.
