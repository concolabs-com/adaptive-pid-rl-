# Chapter 7 — Conclusion and Future Work

## 7.1 Conclusions

This thesis studied online PID gain scheduling under hidden dynamics
parameters, framed as a hidden-parameter MDP and investigated through a
teacher–student comparison of a context-aware agent against a from-scratch
blind agent, on a wheeled vehicle and an inverted pendulum, against fair
classical baselines.

**RQ1 (feasibility).** Learned gain scheduling reliably controls the plant
across a wide range of unknown, randomized dynamics, with 100% hold success
across all scenarios and seeds and generalization to out-of-distribution mass
and actuator conditions.

**RQ2 (privileged context).** Observing the hidden parameters yields a modest
settling-time advantage (1–13%) that is largest when the plant is fast and
vanishes when it is actuator-limited. The original uniform "+14.5%" advantage
was shown to be a measurement artifact; the corrected, regime-dependent result
is the honest answer, and implies that the cost of blindness is small precisely
where the plant is hardest to move quickly.

**RQ3 (memory).** Frame-stack depth is nearly irrelevant — even a near-
memoryless policy succeeds, because the identifiable dynamics are largely
exposed by the instantaneous observation — while a recurrent (GRU) policy is
the fastest variant, albeit with a recurrence-versus-capacity confound.

**RQ4 (representation).** The blind agent's representation encodes the hidden
parameters in a double dissociation matching the plant physics: mass during the
acceleration phase, actuator strength during cruise, friction not at all. The
agent performs genuine implicit system identification.

**Classical comparison and scope of the RL advantage.** A PID with textbook
anti-windup solves the windup-prone task that naive PID fails, so learned
control wins neither on windup nor on transient shock rejection (mid-episode
authority loss defeats all controllers equally). Its distinctive value is
per-episode initial adaptation to unknown dynamics, demonstrated most sharply on
the unstable pendulum, where fixed gains fail the hard corners and the learned
agents do not. MRAC, even with a feasible reference model, fails entirely.

**Methodological contribution.** The audit that produced these corrected
results — uncovering a unit error, a context bug, a wrong-target measurement, a
task-aware environment aid, and a non-discriminative randomization axis, each
capable of inverting a conclusion — is itself a contribution: a demonstration
that simulation aids, units, and observation plumbing must be audited before
adaptive controllers are benchmarked.

## 7.2 Future Work

**Width-matched memory ablation.** Resolve the RQ3b confound by comparing the
GRU against an MLP of equal parameter count, isolating recurrence from capacity.

**A plant where identification is hard.** The blindness penalty was small
because identification here is easy and largely instantaneous. A plant with a
slowly-revealed or weakly-excited hidden parameter (e.g. viscous/slip-dependent
friction, or a delayed actuator fault) would stress implicit identification and
is the natural setting to see context matter more — and to make the third
randomization axis meaningful.

**Approach-phase gain scheduling.** Characterize the gains during the approach
and braking phases, where scheduling (as opposed to the near-invariant hold-
phase operating point) actually occurs, to describe what the learned schedule
does.

**Distillation and explicit context inference.** Compare the from-scratch blind
student against an RMA-style distilled student and against an explicit context
encoder (PEARL/UP-OSI), to separate the cost of blindness from the benefit of a
particular inference scheme.

**Broader transfer and more seeds.** Extend the pendulum study to more plants
and seeds, and add bootstrap-backed significance throughout, to strengthen the
generality and statistical claims.

**Toward hardware.** The sim-to-real gap is unaddressed here. The integral-reset
finding suggests the first step is an honest sim-to-sim audit — verifying that
no simulation convenience is carrying the result — before any hardware transfer
is attempted.

## 7.3 Closing Remark

The most durable outcome of this work is not that reinforcement learning can
schedule PID gains — it can — but a sharpened account of *when that matters*:
not for windup, which classical anti-windup handles; not for sudden authority
loss, which nothing handles by gains alone; but for committing quickly to good
gains on an unknown, possibly unstable plant. Reaching that account required
discarding an attractive but artifactual first result, and that discipline —
auditing the benchmark before trusting the comparison — is offered as the
thesis's broadest contribution.
