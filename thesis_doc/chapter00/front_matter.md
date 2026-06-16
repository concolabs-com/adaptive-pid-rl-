# Front Matter

> Assembled separately from the chapters; place before Chapter 1 in the
> compiled thesis. Replace bracketed institutional fields before submission.

---

## Title page

**Learning to Schedule PID Gains Under Hidden Dynamics Parameters**
*A Teacher–Student Study of Implicit System Identification*

[Author] · [Degree: MSc Data Science] · [Department] · [University]
Supervisor: [Name] · [Month Year]

---

## Abstract

Proportional–integral–derivative (PID) controllers dominate industrial motion
control but are tuned for a single operating regime; when a system's physical
parameters — mass, friction, actuator strength — are hidden and vary between or
within deployments, fixed gains degrade. This thesis studies whether a
model-free reinforcement-learning agent can schedule PID gains online for such
a system, framed as a **hidden-parameter Markov decision process (HiP-MDP)**.
The central comparison is a **teacher–student** one: a context-aware agent that
observes the hidden parameters against a blind agent that must infer them from
interaction history (implicit system identification), trained identically apart
from the observation.

On a simulated two-wheeled vehicle with randomized mass, friction, and actuator
strength, both agents learn reliable control — 100% hold success across eight
scenarios and five seeds, including out-of-distribution conditions. The
privileged-context advantage is modest and regime-dependent (1–13% faster
settling, largest when the plant is fast, vanishing when it is
actuator-limited), correcting an earlier result that an evaluation artifact had
inflated to a uniform 14.5%. A stack-depth ablation shows temporal memory is
nearly irrelevant — a near-memoryless policy succeeds — while a recurrent
policy is the fastest variant. A probing analysis reveals a double dissociation
in the blind agent's representation that matches the plant physics: mass is
decodable during acceleration, actuator strength during cruise, friction not at
all. Against fair classical baselines, a PID with textbook anti-windup solves
the windup-prone task that naive PID fails, and model-reference adaptive control
fails even under a feasible reference model; the learned controllers' distinctive
value is per-episode initial adaptation rather than windup handling or
mid-episode disturbance rejection — demonstrated most sharply on an unstable
inverted-pendulum transfer task where fixed gains fail the hard dynamics
corners. The work also documents and corrects a set of measurement and
environment-design artifacts that had compromised the initial results, arguing
that such auditing is a necessary part of benchmarking adaptive control in
simulation.

**Keywords:** reinforcement learning, PID gain scheduling, hidden-parameter
MDP, domain randomization, system identification, teacher–student learning,
anti-windup, PPO.

---

## Acknowledgements

[To be written by the author.]

---

## Notation and Abbreviations

| Symbol | Meaning |
|--------|---------|
| $\psi = (m,\mu,\kappa)$ | hidden parameter vector: mass, friction, actuator strength |
| $e(t)$ | tracking error, target − position |
| $K_p, K_i, K_d$ | PID proportional / integral / derivative gains |
| $a \in [-1,1]^3$ | RL action: normalized gain multipliers |
| $v_{\max}$ | actuator-limited terminal speed, $\propto \kappa/b$ |
| $\Delta t$ | control period, 0.02 s (RL/eval); physics step 0.002 s |
| $b$ | viscous joint damping |
| $\gamma, \lambda, \epsilon$ | discount, GAE parameter, PPO clip |
| $A_t$ | advantage estimate |
| $T_t$ | anti-windup back-calculation tracking time constant |
| $\tau_m$ | MRAC reference-model time constant |
| $R^2$ | probe coefficient of determination |

| Abbrev. | Expansion |
|---------|-----------|
| PID / PI | proportional–integral–(derivative) controller |
| MDP / POMDP | (partially observable) Markov decision process |
| HiP-MDP | hidden-parameter MDP |
| RL | reinforcement learning |
| PPO | proximal policy optimization |
| GAE | generalized advantage estimation |
| TRPO | trust-region policy optimization |
| MRAC | model-reference adaptive control |
| GRU | gated recurrent unit |
| RMA | rapid motor adaptation |
| OOD | out-of-distribution |
| IAE | integral of absolute error |
| SIMC | Skogestad internal-model-control tuning |
| FOPDT | first-order-plus-dead-time |

---

## Contents (outline)

1. Introduction
2. Background and Related Work
3. System Design and Methodology
4. Experimental Setup
5. Results and Analysis
6. Discussion and Limitations
7. Conclusion and Future Work
- Appendices A–D
- References

*(Generate the paginated table of contents, list of figures, and list of
tables from the typesetting tool — LaTeX `\tableofcontents` / `\listoffigures`
/ `\listoftables` — once the chapters are compiled.)*

---

## List of Figures (to populate on compile)

- Fig 3.1 Two-loop control architecture
- Fig 3.2 Car task environment schematic
- Fig 3.3 Curriculum schedule
- Fig 3.4 Actor–critic network architecture
- Fig 5.x Actuator sweep (settling/overshoot/success vs strength)
- Fig 5.x Probe R² over episode time (mass/actuator/friction)
- Fig 5.x Learned gains vs actuator strength
- Fig 5.x Mid-approach shock trajectories
- Fig 5.x Mass sweep across controllers

## List of Tables (to populate on compile)

- Tab 3.1 PPO hyperparameters
- Tab 5.1 Context vs blind settling (5 seeds)
- Tab 5.2 Stack-depth ablation
- Tab 5.3 Memory-mechanism comparison
- Tab 5.4 No-reset classical baselines
- Tab 5.5 Shock-probe recovery
- Tab 5.6 Pendulum survival
