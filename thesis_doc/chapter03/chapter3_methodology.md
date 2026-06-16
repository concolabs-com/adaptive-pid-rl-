# Chapter 3 — System Design and Methodology

This chapter describes the two simulated plants, the two-loop control
architecture shared across all controllers, the gain parameterization and
reward, the three domain-randomization axes, the curriculum, and the training
protocol. Two learned agents are studied: a **Context-Aware Agent** (the
teacher, observing the hidden parameters) and a **Blind Agent** (the student,
inferring them from history). Classical baselines — fixed-gain PID, anti-windup
PID, and MRAC — operate in the same gain-action space. Section 3.7 describes
the `brake_integral_reset` environment aid whose role as a confound is analysed
in Chapter 5.

---

## 3.1 Two-Loop Control Architecture

All controllers share a cascaded two-loop structure. An **outer loop** runs at
50 Hz: each step it observes the system and outputs a 3-vector of normalized
gain adjustments. An **inner PID loop** runs at 500 Hz (MuJoCo physics
timestep $\Delta t_{\text{phys}} = 0.002$ s, with `frame_skip` = 10 so the
control period is $\Delta t = 0.02$ s): each physics sub-step the PID computes
the actuator command from the current gains and error. The learned policy thus
*schedules* the gains of a classical controller rather than emitting torque
directly. For the learned agents the outer loop is the RL policy; for the
classical baselines the outer loop emits a constant action (fixed gains) or the
MRAC adaptation law.

A measurement note that pervades the results: position is logged once per
**control** step, so settling time and IAE must be computed with
$\Delta t = 0.02$ s, not the physics timestep $0.002$ s. Using the physics
timestep understates all times by exactly 10× — an error identified in the
audit (Chapter 5) and corrected throughout.

## 3.2 Primary Plant: Two-Wheeled Vehicle

The primary plant is a two-wheeled MuJoCo vehicle driven by two wheel motors
with command range $[-1,1]$ and joint actuator-force limit $\pm 0.5$ N·m, with
viscous joint damping $b = 0.03$. The task: drive from $x=0$ to a target
(nominally 5 m, randomized 3–7 m in training) and hold within $\pm 0.05$ m for
25 consecutive control steps.

**Terminal-speed analysis (why the task behaves as it does).** Under wheel
torque the longitudinal dynamics are approximately

$$ m\,\ddot{x} = \frac{2\,\kappa\,\tau(u)}{r_w} - \frac{2b}{r_w^2}\,\dot{x}, $$

where $\kappa$ scales actuator strength and $r_w$ is the wheel radius. At
saturated command the steady cruise speed is $v_{\max} \propto \kappa /
b$ — **independent of mass**. With the nominal parameters $v_{\max}\approx
0.5$ m/s. Two consequences follow and shape every result. First, settling time
for a multi-metre drive is dominated by travel time $\approx d/v_{\max}$, so
**mass barely affects settling** (a 10× mass change moves fixed-PID settling
~11%, §5.x) whereas **actuator strength does** ($v_{\max}\propto\kappa$ moves
it 3.7×). Second, the actuator saturates throughout the cruise, which is why
integral windup (§2.2) is the dominant failure mode of a naive PID and why the
identifiability structure of §2.4.4 holds.

**Gain parameterization.** The policy action $a\in[-1,1]^3$ maps to gains
around a base:

$$ K_x = \mathrm{clip}\big(K_{x,\text{base}} + \Delta K_x\, a_x,\ K_{x,\min}, K_{x,\max}\big),\quad x\in\{p,i,d\}, $$

with $K_{p,\text{base}}=1.8,\ K_{i,\text{base}}=0.7,\ K_{d,\text{base}}=0.5$
and deltas $\Delta K_p=1.0,\ \Delta K_i=0.6,\ \Delta K_d=2.0$, giving effective
ranges $K_p\in[0.8,2.8]$, $K_i\in[0.1,1.3]$, $K_d\in[0.0,2.5]$.

**Observation.** Per control step the raw observation is 9-dimensional:
position, velocity, error, the three previous normalized gain actions, and the
three context scales (mass, friction, actuator). The Context-Aware Agent
receives all 9 dimensions; the Blind Agent receives the first 6 (context
dropped). Both stack the last 10 observations, giving 90-dim and 60-dim policy
inputs respectively.

**Reward** (per control step):

$$ r = 5\,\Delta x \;-\; 0.75\,|e| \;-\; 0.12\,\dot{x}^2 \;-\; 2.0\,\max(0, x-x_{\text{target}}), $$

with a progress term, a distance penalty, a velocity penalty, and a direct
overshoot penalty. Inside the braking zone ($|e| < 2$ m) the progress and
distance terms are zeroed and a dense deceleration bonus ($+10\,\Delta(-\dot
x)$ while slowing) plus a quadratic velocity penalty shape a controlled stop; a
terminal bonus of $+80$ is given on successful hold completion. The shaping was
arrived at iteratively (governor and hard-overshoot "cliff" variants were tried
and dropped — see §3.6); the final form rewards approach, then deceleration,
then a held stop, without a hard termination that was found to starve
exploration.

## 3.3 Domain Randomization — Three Hidden Axes

Each episode samples a hidden parameter vector $\psi = (m,\mu,\kappa)$:

| Axis | Range | Dynamical effect |
|------|-------|------------------|
| Mass $m$ | $[5,20]$ kg | inertia; affects transient ($\ddot x = F/m$) only |
| Friction $\mu$ | $[0.1,2.0]$ | **inert** — rolling contact, no slip (see below) |
| Actuator strength $\kappa$ | $[0.6,1.4]$ | scales $v_{\max}\propto\kappa$ — the discriminative axis |

The actuator axis scales the motor gain and the joint force limit together, so
the whole torque envelope (and hence terminal speed) scales by $\kappa$. It was
added after the audit found that mass-only randomization left the task nearly
non-discriminative (§5.x). **Friction is retained but documented as inert**:
because the wheels roll without slipping at these operating points, the sliding-
friction coefficient does not enter the translational dynamics; the probing
analysis (§5.x) confirms it is at best weakly identifiable.

**Within-episode variation.** A **mid-episode disturbance** fires at a random
step in $[120,220]$ (2.4–4.4 s), multiplying mass by $\in[0.9,1.3]$, friction
by $\in[0.5,1.4]$, and (in v6) actuator strength by $\in[0.85,1.15]$. A
**position friction patch** reduces friction to 35% of nominal over
$x\in[1.5,2.4]$ m. With the corrected timeline the disturbance window overlaps
the approach phase (settling is ~12–15 s), so the disturbance genuinely
perturbs the controller mid-manoeuvre — contrary to the original draft's claim
that it fired post-settling.

## 3.4 Policy Architecture and PPO

The policy is an actor–critic MLP: two hidden layers of 64 units with Tanh
activations and orthogonal initialization, separate actor and critic heads, a
3-dim Gaussian action with a learned log-std, tanh-squashed to $[-1,1]$. The
recurrent variant (RQ3b) replaces the shared trunk with a 128-unit GRU and uses
a stack size of 1. Training uses PPO (§2.4.2) with the hyperparameters in
Table 3.1 (full list in Appendix A).

| Parameter | Value | Parameter | Value |
|-----------|-------|-----------|-------|
| Learning rate | $3\times10^{-4}$ (linear decay) | Minibatch | 64 |
| $\gamma$ | 0.99 | Update epochs | 10 |
| $\lambda$ (GAE) | 0.95 | Rollout steps | 2048 / env |
| Clip $\epsilon$ | 0.2 | Parallel envs | 4 |
| Value coef | 0.5 | Total steps | 1,000,000 |
| Entropy coef | 0.0 | Frame stack | 10 |

*Table 3.1 — PPO hyperparameters (car).* Training seeds: 7, 21, 42, 84, 123.

## 3.5 Curriculum Learning

Target distance grows over four equal phases of training:

| Phase | Steps | Target range |
|-------|-------|--------------|
| 1 | 0–250k | 1–3 m |
| 2 | 250k–500k | 1–5 m |
| 3 | 500k–750k | 1–7 m |
| 4 | 750k–1M | 1–10 m |

Short targets first let the agent learn the held-stop behaviour before facing
the long approaches where windup pressure is greatest. The disturbance
magnitude is held fixed across phases.

## 3.6 Classical Baselines (Same Action Space)

**Fixed PID** emits the constant action $[0,0,0]$ (base gains $K_p=1.8,
K_i=0.7, K_d=0.5$) — the best a correctly pre-tuned non-adaptive controller can
do. **Anti-Windup PID** adds back-calculation ($T_t=1$ s) or conditional
integration to the same fixed gains; this is the *fair* classical comparator
(§2.2). **MRAC** replaces the constant action with the MIT-rule adaptation law
(§2.3) under a feasible reference model ($\tau_m\ge 10$ s, consistent with
$v_{\max}=0.5$ m/s) and a normalized update. All three run through the identical
two-loop simulation, so differences reflect the control strategy, not the
harness.

## 3.7 The Brake Integral Reset Aid

During development the environment included a `brake_integral_reset` that zeroes
the PID integrator on entry to the braking zone ($|e| < 2$ m). It is, in effect,
a *task-aware* conditional integration: it consults distance-to-target rather
than actuator saturation. It is active by default for all controllers. Its
effect — equalizing performance by removing windup at the critical moment — is
the confound analysed in §5.x: with it, naive and anti-windup PID are
indistinguishable; without it, only anti-windup (or the learned agents) cope.

## 3.8 Transfer Plant: Inverted Pendulum

To test generality the same architecture is applied to a MuJoCo inverted
pendulum (cart-pole) balance task. The hidden parameters are pole-mass scale
$[0.5,2.5]$ and actuator-gear scale $[0.6,1.4]$. A fixed **cascade outer loop**
shapes the angle setpoint from cart position, $\theta_{\text{ref}} =
\mathrm{clip}(k_x x + k_{\dot x}\dot x, \pm\theta_{\max})$, and the scheduled
PID stabilizes the pole angle about $\theta_{\text{ref}}$; the RL action
multiplies the angle-PID gains, exactly as for the car. The reward rewards
upright balance and centred cart and penalizes failure (pole or cart leaving
bounds). This plant is *unstable*, so the value of adaptation is expected to be
larger than on the self-stabilizing car — borne out in §5.x.
