# Chapter 1 — Introduction

## 1.1 Motivation

The proportional–integral–derivative (PID) controller is the workhorse of
industrial motion control. It is simple, interpretable, requires no model of
the plant, and has a track record spanning a century of practice; surveys
consistently report that the overwhelming majority of control loops in service
are PID or a close variant [^astrom1995]. Its central weakness is equally
well known: a PID controller is tuned for one operating regime. Its gains
$K_p, K_i, K_d$ encode assumptions about the plant — its inertia, its damping,
the authority of its actuator — and when those assumptions are violated the
controller degrades. A vehicle that is heavier than the gains were tuned for
overshoots its target; a motor that has weakened with wear becomes sluggish;
a surface that has changed underfoot upsets a loop tuned for a different one.

These are not hypothetical edge cases; they are the normal operating envelope
of real motion systems. A warehouse automated-guided vehicle carries anything
from an empty tote to a full pallet, a several-fold payload-mass swing within a
single shift. A delivery drone's effective actuator authority falls as its
battery sags and its propellers erode. A robot manipulator's effective inertia
at the wrist depends on the tool and workpiece it happens to be holding. An
electric drive's torque constant drifts with temperature over a duty cycle. In
each case the plant the controller faces at run time differs — sometimes
severely — from the plant it was tuned against, and in each case the deviating
quantity is typically *not instrumented*: there is no payload-mass sensor on the
AGV, no direct actuator-health signal on the drone. The control engineer's
classical options are to tune conservatively for the worst case (sacrificing
performance in the common case), to add the missing sensor (cost, weight,
failure modes), or to build a gain schedule indexed by a measured proxy (which
presumes the proxy exists and the schedule was designed in advance). This thesis
asks what a learning approach can offer when none of those is attractive.

In many deployments the operating regime is not merely uncertain but *hidden*:
the controller has no sensor for payload mass, for actuator health, or for the
properties of the surface it drives on. Re-tuning by hand for each condition is
infeasible when the condition is unknown at design time and changes during
operation. The classical remedy, **gain scheduling**, interpolates between
pre-computed gain sets indexed by a measured scheduling variable — but it
presupposes that the scheduling variable *can* be measured and that the
schedule has been designed in advance. **Model-reference adaptive control
(MRAC)** removes the lookup table by adapting gains online toward a reference
model, but it assumes a plant structure that the adaptation law can exploit,
and it is fragile on nonlinear, input-saturated systems (Chapter 5 documents
its failure here).

This thesis asks whether **reinforcement learning (RL)** offers a third route:
a controller that learns, from interaction alone, to set PID gains in real time
for a plant whose physical parameters are hidden and vary from one deployment
to the next. The agent never receives a plant model and, in its most practical
form, never measures the hidden parameters; it must infer whatever it needs
from the closed-loop trajectory and act on that inference through the gains of
an ordinary PID loop.

## 1.2 Problem framing: a hidden-parameter MDP

The setting is naturally formalized as a **hidden-parameter Markov decision
process (HiP-MDP)** [^doshivelez2016]: a family of control problems sharing a
state space, an action space, and a reward, but differing in their transition
dynamics through a latent parameter vector $\psi$ — here the vehicle mass,
ground friction, and actuator strength. Each episode draws a $\psi$ that is
hidden from the controller. Training across the distribution of $\psi$ —
**domain randomization** [^tobin2017] — forces a single policy to perform
across the whole family rather than overfitting one operating point.

Within this frame the thesis studies a deliberately sharp contrast, borrowed
from the **privileged-information** or **teacher–student** paradigm that
underlies recent learned control [^kumar2021]:

- a **context-aware agent** (the *teacher*) that observes $\psi$ directly, and
- a **blind agent** (the *student*) that observes only the ordinary control
  signals — position, velocity, error, and its own recent gains — and must
  therefore perform **implicit system identification**: inferring the hidden
  dynamics from the shape of the trajectory it produces.

The blind agent faces a partially observable problem; it is given a short
window of stacked past observations as a finite-memory surrogate for the belief
over $\psi$. The teacher, given $\psi$, faces an ordinary MDP. Comparing the
two isolates one question precisely — *what does it cost to infer the dynamics
rather than be told them?* — because the two agents are identical in every
other respect: same architecture, same reward, same training budget, same
curriculum. Unlike RMA, the student is trained from scratch rather than
distilled from the teacher, so the comparison measures the cost of blindness
itself, not the quality of a distillation scheme.

A second deliberate choice is the **action space**. The agent does not emit
motor torques; it emits multipliers on the gains of a classical PID loop that
computes the torque. The learned policy modulates an interpretable controller
rather than replacing it. This keeps the classical baselines — fixed-gain PID,
anti-windup PID, MRAC — directly comparable in the same action space, and it
keeps the learned behaviour legible: one can read off the gain schedule the
agent has learned and ask whether it corresponds to anything a control engineer
would recognize.

## 1.3 Research questions

The thesis is organized around four questions.

**RQ1 (Feasibility).** Can an RL agent learn online PID gain scheduling that
reliably drives the plant to its target and holds it, across a wide range of
unknown, randomized dynamics — including conditions outside the training
distribution?

**RQ2 (Privileged context).** Does observing the hidden parameters (the
context-aware teacher) improve control over inferring them from interaction
history (the blind student), and under what conditions does the advantage
appear or vanish?

**RQ3 (Memory).** How much temporal context does the blind agent require —
how deep a history, and does a learned recurrent memory outperform a fixed
window?

**RQ4 (Representation).** Does the blind agent's policy actually *encode* the
hidden parameters in its internal representation, and when during an episode
does each parameter become decodable?

RQ1 and RQ2 establish that the approach works and quantify the cost of
blindness. RQ3 and RQ4 turn from *whether* to *how*: they probe the mechanism
of the implicit identification that makes the blind agent possible.

## 1.4 A note on scientific method

The experimental work behind this thesis was audited and substantially
re-run after the first set of results proved to rest on measurement artifacts.
The audit is not hidden in an appendix; it is treated as a contribution.
Four issues were identified and corrected, each instructive in its own right:
a unit error that made all settling times appear ten times too fast; an
observation-plumbing bug that fed the context-aware agent a constant,
out-of-distribution context during evaluation and so manufactured an apparent
advantage; a baseline measured against the wrong target that made a classical
controller look catastrophically worse than it is; and a task whose original
randomization axis (mass) barely affected the dynamics, so that no controller
could be distinguished from another. Section 1.5 lists the corrected
contributions; Chapter 5 reports the corrected results; Chapter 6 reflects on
what the episode says about benchmarking learned control. The methodological
lesson — that simulation aids and measurement conventions must be audited
before adaptive controllers are compared — is one of the more transferable
findings of the work.

## 1.5 Contributions

1. **A HiP-MDP formulation of PID gain scheduling and a teacher–student study
   of it.** The thesis frames online gain scheduling under hidden dynamics as a
   HiP-MDP and reports a controlled comparison of a context-aware teacher
   against a from-scratch blind student, holding all else equal. Both achieve
   reliable control across a 4:1 mass range, a wide actuator-strength range,
   and out-of-distribution conditions; the privileged-context advantage is
   shown to be modest and regime-dependent rather than uniform.

2. **A mechanistic account of implicit system identification.** Through a
   stack-depth and recurrence ablation (RQ3) and a probing analysis of the
   learned representation (RQ4), the thesis shows *what* the blind agent infers
   and *when*: mass becomes decodable during acceleration and actuator strength
   during cruise, a double dissociation that matches the plant's physics. It
   also shows the limits — frame-stack depth barely matters, gains are closer
   to a robust operating point than a steep schedule, and mid-episode losses of
   actuator authority cannot be rescued by gain changes.

3. **A fair classical-baseline study and a methodological audit.** The thesis
   establishes the *fair* classical comparator — a PID with textbook
   anti-windup, which solves the hardest version of the task that naive PID
   fails — and reports the failure of MRAC under a feasible reference model. It
   documents and corrects the measurement artifacts that compromised the
   initial results, arguing that such auditing is a necessary part of
   benchmarking adaptive control in simulation.

4. **A cross-plant transfer result.** The same architecture, applied to an
   unstable inverted-pendulum balance task with randomized pole mass and
   actuator gear, again learns to schedule gains and clearly outperforms fixed
   gains on the hard dynamics corners — evidence that the framework is not
   specific to the wheeled-vehicle plant.

## 1.6 Key findings in brief

Both agents reach 100% hold success across all evaluated conditions and seeds.
The context-aware teacher settles between 1% and 13% faster than the blind
student, the gap widening when the plant is fast and closing when it is
actuator-limited. Frame-stack depth has no measurable effect, but a recurrent
(GRU) blind agent is the fastest variant of all. Probes recover mass from the
acceleration phase and actuator strength from the cruise phase of the blind
agent's own representation. A PID with anti-windup matches the learned
controllers on the windup-prone variant of the task that naive PID fails, while
MRAC fails entirely; the learned controllers' distinctive value is initial,
per-episode adaptation to unknown dynamics, demonstrated most sharply on the
unstable pendulum where fixed gains fail the hard corners outright.

## 1.7 Scope

The work is conducted entirely in simulation (MuJoCo). Sim-to-real transfer is
out of scope: no hardware experiments are performed and no claim of
deployability on a physical system is made. The contribution is an analysis of
learned gain scheduling as a control and machine-learning problem — its
feasibility, the value of privileged information, the mechanism of implicit
identification, and an honest accounting of where it does and does not help.

## 1.8 Thesis outline

Chapter 2 develops the background: PID control and its tuning, integral windup
and anti-windup, classical adaptive control, the RL and PPO machinery used
here, and the HiP-MDP and teacher–student literature the work builds on.
Chapter 3 describes the two simulated plants, the two-loop control
architecture, the gain parameterization, the reward, the domain-randomization
axes, and the curriculum. Chapter 4 sets out the evaluation protocol, the
scenario suite, the agents compared, and the statistical methodology. Chapter 5
reports the results question by question (RQ1–RQ4) together with the classical
baselines, the discriminativeness analysis, the disturbance probe, and the
pendulum transfer. Chapter 6 interprets the findings, frames the audit as a
methodological contribution, and states the limitations. Chapter 7 concludes
and outlines future work.

[^astrom1995]: Åström, K. J. and Hägglund, T. (1995). *PID Controllers:
Theory, Design, and Tuning*, 2nd ed. ISA. (Add page/edition reference for the
"95% of loops" statistic.)
[^doshivelez2016]: Doshi-Velez, F. and Konidaris, G. (2016). Hidden Parameter
Markov Decision Processes. *IJCAI*.
[^tobin2017]: Tobin, J. et al. (2017). Domain Randomization for Transferring
Deep Neural Networks from Simulation to the Real World. *IROS*.
[^kumar2021]: Kumar, A., Fu, Z., Pathak, D. and Malik, J. (2021). RMA: Rapid
Motor Adaptation for Legged Robots. *RSS*.
