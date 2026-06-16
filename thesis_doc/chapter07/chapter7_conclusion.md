# Chapter 7 — Conclusion and Future Work

## 7.1 Conclusions

This thesis studied online PID gain scheduling under hidden dynamics
parameters, framed as a hidden-parameter MDP and investigated through a
teacher–student comparison of a context-aware agent against a from-scratch
blind agent, on a wheeled vehicle and an inverted pendulum, against fair
classical baselines.

The overarching question posed in Chapter 1 was whether a model-free agent can
learn to schedule PID gains for a plant whose physics are hidden, and what it
costs to infer those physics rather than be told them. The answer is a
qualified yes with a precise account of the qualification. An agent can learn
the task and generalize beyond its training distribution; inferring the hidden
parameters rather than observing them costs little, because on this plant the
parameters are largely exposed instantaneously; and the learned controller's
advantage over classical control is real but *narrow* — it is an advantage in
per-episode adaptation to unknown dynamics, not in any of the things (windup
rejection, disturbance recovery) a practitioner might first assume. The four
research questions sharpen this into specific, evidenced claims.

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

Six directions follow directly from the limitations (§6.3) and the threats to
validity (§6.5); they are ordered roughly by how cheaply they would strengthen
the central claims.

**Width-matched memory ablation.** The clearest immediate gap is the RQ3b
recurrence-versus-capacity confound: the GRU (128 hidden units) beat the
frame-stacking MLP (2×64), but the comparison does not separate the benefit of
recurrent state from the benefit of extra parameters. Re-running the GRU
against an MLP of matched parameter count — and, symmetrically, a frame-stacking
MLP widened to the GRU's capacity — would isolate which factor carries the
speed advantage. This is a one-experiment fix and would settle whether
"recurrence helps" is a claim the thesis can make.

**A plant where identification is hard.** The small blindness penalty (RQ2) and
the flat stack-depth curve (RQ3) both trace to the same cause: on this plant the
hidden parameters are *easy* to identify, exposed almost instantaneously by
velocity and the previous gains. The natural way to make the teacher–student
gap large — and to make implicit identification genuinely difficult — is a plant
where a parameter is revealed only slowly or weakly: viscous or slip-dependent
friction (which would also make the inert third axis meaningful, addressing L1),
a delayed or intermittent actuator fault, or a parameter that only manifests
under specific manoeuvres. The probing methodology of RQ4 transfers directly and
would predict, before training, which parameters a blind agent *can* recover.

**Approach-phase gain scheduling.** The gain analysis (§5.9) examined the hold
phase and found near-invariant gains — a robust operating point, not a steep
schedule. But scheduling, if it exists, lives in the approach and braking
phases where the controller actively fights the dynamics. Characterizing the
gain trajectories there (phase-aligned across episodes, regressed against the
hidden parameter) would either reveal a genuine schedule the hold-phase analysis
missed or confirm that the learned policy is fundamentally a robust regulator —
either outcome sharpens the thesis's most hedged claim.

**Distillation and explicit context inference.** The blind student here is
trained from scratch, which isolates the *observation regime* but conflates it
with the *training scheme*. Comparing against an RMA-style student distilled
from the teacher's latent embedding, and against an explicit context encoder
(PEARL, UP-OSI) that regresses the parameters, would separate the cost of
blindness from the benefit of a particular inference architecture — and would
test whether distillation closes even the small gap measured here.

**Broader transfer, more seeds, and powered statistics.** The pendulum is a
two-seed transfer *demonstration*; promoting it to a second full study (five
seeds, the complete scenario suite, bootstrap-backed significance) and adding a
third, structurally different plant would turn the generality claim from
suggestive to demonstrated. The same five-to-more-seeds increase on the car
would tighten the wide RQ2 confidence intervals enough to state effect sizes,
not just directions.

**Toward hardware.** The sim-to-real gap is unaddressed by design (§1.7), and
the integral-reset episode is a pointed reminder that even simulation
conveniences can carry a result. The responsible first step is therefore not a
hardware port but an honest *sim-to-sim* audit: retrain without every
task-aware aid, verify the policy still works when no convenience is doing the
hard part, and only then characterize the domain gap to a physical vehicle.
This ordering — audit the benchmark, then transfer — is the practical
expression of the thesis's methodological contribution.

## 7.3 Closing Remark

The most durable outcome of this work is not that reinforcement learning can
schedule PID gains — it can — but a sharpened account of *when that matters*:
not for windup, which classical anti-windup handles; not for sudden authority
loss, which nothing handles by gains alone; but for committing quickly to good
gains on an unknown, possibly unstable plant. Reaching that account required
discarding an attractive but artifactual first result, and that discipline —
auditing the benchmark before trusting the comparison — is offered as the
thesis's broadest contribution.
