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

where $\kappa$ scales actuator strength and $r_w$ is the wheel radius. During
the cruise the command saturates at $u=u_{\max}$, so the forcing term
$F \equiv 2\kappa\tau(u_{\max})/r_w$ is constant and the equation is a linear
first-order ODE in the velocity $v=\dot x$:

$$ m\,\dot v + \frac{2b}{r_w^2}\,v = F, \qquad v(t) = v_{\max}\Big(1 - e^{-t/\tau_v}\Big), $$

with **terminal speed** and **velocity time constant**

$$ v_{\max} = \frac{F\,r_w^2}{2b} = \frac{\kappa\,\tau(u_{\max})\,r_w}{b}, \qquad \tau_v = \frac{m\,r_w^2}{2b}. $$

Two facts fall out and shape every result. First, $v_{\max}\propto\kappa/b$ is
**independent of mass** — mass appears only in the time constant $\tau_v$,
i.e. in how fast the vehicle *reaches* cruise, not how fast it cruises. With the
nominal parameters $v_{\max}\approx 0.5$ m/s and $\tau_v$ on the order of a few
hundred milliseconds, so for a multi-metre drive the constant-velocity cruise
dominates the trip time. Settling time is therefore $\approx d/v_{\max}$ plus a
mass-dependent transient of order $\tau_v$: a 10× mass change moves fixed-PID
settling only ~11% (§5.7), whereas an actuator change scales $v_{\max}$
directly and moves it 3.7×. This is the quantitative reason mass is a poor
discriminator and actuator strength a good one — and the reason the original
mass-only randomization (F9) could not separate controllers. Second, because
the command saturates throughout the cruise, integral windup (§2.2) is the
dominant failure mode of a naive PID, and the parameter that is excited in each
phase — mass in the acceleration transient ($\tau_v\propto m$), actuator
strength in the cruise ($v_{\max}\propto\kappa$) — is exactly what the probing
analysis recovers (§2.4.5, §5.5).

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
terminal bonus of $+80$ is given on successful hold completion.

Each term answers a specific failure mode observed during development
(Table 3.x):

| Term | Form | Purpose / failure it prevents |
|------|------|-------------------------------|
| Progress | $+5\,\Delta x$ | dense signal to move toward the target; without it the sparse hold bonus is almost never reached by exploration |
| Distance | $-0.75\,|e|$ | breaks ties among slow approaches; rewards getting *and staying* close |
| Velocity | $-0.12\,\dot x^2$ | discourages a fast fly-by that cannot be braked; quadratic so it bites hardest at high speed |
| Overshoot | $-2.0\max(0,x-x_t)$ | direct penalty for passing the target — the single most damaging error on this windup-prone plant |
| Decel bonus | $+10\,\Delta(-\dot x)$ in zone | a *dense* braking reward so deceleration is reinforced step-by-step rather than only through the terminal bonus |
| Terminal | $+80$ on hold | the actual task objective; large enough to dominate once reachable |

Two design choices proved important. First, **zeroing progress and distance
inside the braking zone** removes the perverse incentive to keep inching
forward for progress reward when the agent should be stopping; in the zone the
only gradients are the deceleration bonus and the velocity penalty, which
together specify "slow down and hold". Second, the shaping was arrived at
iteratively: earlier variants used a speed *governor* (a hard cap that masked
what the policy had learned about braking) and a hard-overshoot *cliff* (a
terminating penalty for passing the target). Both were dropped — the governor
because it confounded the evaluation of learned braking, the cliff because the
hard termination starved exploration (the agent rarely survived long enough to
discover the hold bonus, and the value function collapsed to the cliff penalty
everywhere). The final form rewards approach, then deceleration, then a held
stop, using only soft penalties, and is the `thesis_v4_cliff`/`thesis_v6_hipmdp`
reward; the dropped variants are documented because their failure is itself
informative about reward design on input-saturated plants.

## 3.3 Domain Randomization — Three Hidden Axes

Each episode samples a hidden parameter vector $\psi = (m,\mu,\kappa)$:

| Axis | Range | Dynamical effect |
|------|-------|------------------|
| Mass $m$ | $[5,20]$ kg | inertia; affects transient ($\ddot x = F/m$) only |
| Friction $\mu$ | $[0.1,2.0]$ | **inert** — rolling contact, no slip (see below) |
| Actuator strength $\kappa$ | $[0.6,1.4]$ | scales $v_{\max}\propto\kappa$ — the discriminative axis |

The ranges are chosen to be wide enough to demand adaptation yet within the
plant's physical capability. Mass spans a **4:1 ratio** (5–20 kg), bracketing
the empty-to-loaded swing of a small vehicle; the actuator axis spans
$\pm40\%$ of nominal strength, enough to move terminal speed by the same factor
without rendering the task impossible. The actuator axis scales the motor gain
and the joint force limit *together*, so the whole torque envelope (and hence
the terminal speed $v_{\max}\propto\kappa$ derived in §3.2) scales by $\kappa$;
scaling only one would let the controller saturate against the unscaled limit
and partially mask the change. This axis was added after the audit found that
mass-only randomization left the task nearly non-discriminative (§5.7) — a
direct consequence of $v_{\max}$ being mass-independent. **Friction is retained
but documented as inert**: because the wheels roll without slipping at these
operating points, the Coulomb friction coefficient does not enter the
translational dynamics (it would matter only at the slip boundary, which the
low torques never reach); the probing analysis (§5.5) confirms it is at best
weakly identifiable. Retaining it rather than deleting it keeps continuity with
the pre-audit experiments and provides a built-in *negative control* for the
probing analysis — a parameter that is randomized but should *not* be
decodable, against which the genuine decodability of mass and actuator strength
can be calibrated.

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

The architecture is deliberately **small and standard**. The 2×64 Tanh trunk is
the canonical continuous-control MLP; nothing about the gain-scheduling task
calls for more capacity, and a small network keeps training fast enough to run
five seeds plus ablations on a laptop CPU. The action is a 3-dimensional
diagonal Gaussian whose mean is the network output and whose log-standard-
deviation is a learned state-independent parameter vector; the mean is
tanh-squashed to $[-1,1]$ to respect the action bounds, and at evaluation the
policy is made deterministic by taking the mean. Separate (non-shared) actor and
critic trunks are used because the value function under the heavily shaped
reward has a very different scale and curvature from the policy, and sharing
features was found to couple their optimization unhelpfully. The choice of
**frame stacking** ($k=10$, a 0.2 s window) over recurrence as the *default*
memory is itself a deliberate, testable decision: a fixed window turns the POMDP
into an approximate MDP on a fixed-size input, is trivially parallelizable, and
trains far faster than backpropagation through time — and whether the extra
machinery of recurrence is even warranted is precisely what the RQ3 ablation
(§5.4) measures rather than assumes. The context and blind agents differ in
*exactly one respect* — the presence of the three context dimensions in the
observation — so that any performance gap is attributable to the observation
regime and not to a confounding architectural difference; this single-variable
discipline is what licenses the RQ2 interpretation.

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
a *task-aware* conditional integration: where the textbook clamping of §2.2
consults actuator saturation (information any PID block has), this aid consults
distance-to-target (information that presumes the controller already knows where
the target is) — so it is a stronger, less generalizable intervention than a
standard anti-windup, and it is active by default for every controller,
including the classical baselines.

It is documented here, in the methodology, rather than buried in the results,
because its presence is the single most consequential design decision in the
original experiments and its discovery reframed the entire comparison. Carried
silently, it makes the environment quietly *easy*: it removes the windup that
is a naive PID's only failure mode on this plant, so a correctly pre-tuned fixed
PID looks excellent and the headline "fixed PID is fastest" becomes a statement
about the aid rather than about non-adaptive control. The audit's response was
not to delete it — a real braking-zone integrator reset is a legitimate, if
task-specific, engineering choice — but to make the comparison *robust* to it:
evaluate every controller both with and without it (§5.6), and add the standard,
saturation-based anti-windup PID as the *fair* classical comparator that needs
no task knowledge. With the aid active, naive and anti-windup PID are
indistinguishable (the aid does the anti-windup's job); with it disabled, only
the anti-windup PID — or the learned agents, at a cost — cope. Treating the aid
as an object of study rather than a hidden convenience is the methodological
stance the thesis argues for throughout.

## 3.8 Transfer Plant: Inverted Pendulum

To test generality the same architecture is applied to a MuJoCo inverted
pendulum (cart-pole) balance task. The hidden parameters are pole-mass scale
$[0.5,2.5]$ and actuator-gear scale $[0.6,1.4]$.

**Why a cascade is needed.** A single PID on the pole angle stabilizes the
*pole* but not the *cart*: with the angle held at zero the cart is free to drift
along the rail until it hits the limit and the episode ends. The classical
remedy is a two-level cascade. An inner loop regulates the pole angle $\theta$
to a reference $\theta_{\text{ref}}$; an outer loop sets $\theta_{\text{ref}}$
from the cart state so that, to recover the cart toward the rail centre, the
controller deliberately leans the pole *back* toward centre and lets the
inner loop chase it:

$$ \theta_{\text{ref}} = \mathrm{clip}\big(k_x\,x + k_{\dot x}\,\dot x,\ -\theta_{\max},\ \theta_{\max}\big), $$

with small negative gains $k_x, k_{\dot x} < 0$ (a cart at $+x$ needs a slight
backward lean, $\theta_{\text{ref}}<0$, so the stabilizing push returns it
toward the origin) and a saturation $\theta_{\max}=0.15$ rad that keeps the
commanded lean within the small-angle regime where the inner loop is valid. The
sign and magnitude were verified empirically during base-gain tuning (a
positive-gain cascade drove the cart *off* the rail within 0.1 s of episode
length, confirming the sign). The outer loop is **fixed** for all controllers,
including the RL agents; the learned action multiplies only the inner angle-PID
gains $K_p, K_d$ (with a small $K_i$ range), so the comparison is, as on the
car, purely about how the inner-loop gains are scheduled given the hidden plant.
The reward is $r = 1 - 5\theta^2 - 0.1 x^2 - 0.05\dot x^2$ per step with a
$-20$ penalty on failure (pole past $\pm0.4$ rad or cart past $\pm0.95$ m),
which rewards upright-and-centred balance and sharply punishes a fall.

This plant is *unstable* — left uncontrolled the pole diverges exponentially —
so a poorly matched fixed gain does not merely settle slowly (as on the car) but
falls outright. The value of per-episode adaptation is therefore expected to be
qualitatively larger here, and §5.10 confirms it: fixed gains fail the hard
dynamics corners that the learned agents survive.
