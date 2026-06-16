# Chapter 2 — Background and Related Work

This chapter develops the technical background and situates the thesis in the
literature. Section 2.1 establishes PID control and its tuning. Section 2.2
analyses integral windup and the anti-windup compensation that defines the
*fair* classical baseline. Section 2.3 reviews classical adaptive control —
gain scheduling and MRAC — and what each demands of the designer. Section 2.4
formalizes the learning problem as an MDP, POMDP, and hidden-parameter MDP, and
develops the PPO machinery used throughout. Section 2.5 covers domain
randomization, frame stacking, and the teacher–student / privileged-information
paradigm. Section 2.6 reviews RL for PID tuning and states the gap this thesis
fills.

---

## 2.1 PID Control and Tuning

For a tracking error $e(t) = r(t) - y(t)$ between reference $r$ and measured
output $y$, the PID control law is

$$ u(t) = K_p\,e(t) + K_i \int_0^t e(\tau)\,d\tau + K_d\,\dot{e}(t), $$

with proportional, integral, and derivative gains $K_p, K_i, K_d$. The
proportional term provides restoring effort proportional to the error; the
integral term eliminates steady-state offset by accumulating past error; the
derivative term anticipates, damping oscillation. In the Laplace domain the
controller is $C(s) = K_p + K_i/s + K_d s$, and the closed loop $C(s)P(s)$
around a plant $P(s)$ has dynamics that the gains shape directly.

Tuning chooses the gains, and the major methods illustrate how much plant
knowledge that requires. **Ziegler–Nichols** drives the loop to its stability
limit, records the ultimate gain $K_u$ and oscillation period $T_u$, and reads
the gains from a table ($K_p = 0.6K_u$, $K_i = 1.2K_u/T_u$, $K_d =
0.075K_uT_u$ for classic PID); it needs no model but does need to push the real
plant to sustained oscillation, often unacceptable, and yields aggressive,
oscillation-prone loops. **Internal model control** and the **SIMC** rules of
Skogestad instead start from a first-order-plus-dead-time fit
$P(s) = k\,e^{-\theta s}/(\tau s + 1)$ and choose gains to cancel the plant pole
against a desired closed-loop time constant $\tau_c$, giving (for a PI loop)
$K_p = \tau/\big(k(\tau_c+\theta)\big)$ and $K_i = K_p/\min(\tau, 4(\tau_c+
\theta))$, with $\tau_c$ the single tuning knob trading speed against
robustness. The Stage-1 development of this project used exactly this route — a
step-response FOPDT identification feeding SIMC gains — to obtain the base gains
$K_p=1.8, K_i=0.7$ around which the RL agent later schedules.

All of these share a defining assumption: the plant parameters $(k, \tau,
\theta)$ are **fixed and known** at tuning time. A gain set derived for one
$(k,\tau,\theta)$ is optimal only near that operating point; the IMC pole-
cancellation that makes SIMC clean is exact only when the model matches the
plant. When the plant changes — inertia $k$ with payload, effective $\tau$ with
actuator strength — the cancellation degrades and the loop is mistuned in a
direction the designer did not choose. The classical fix, gain scheduling
(§2.3), restores matching *if* the changing parameter can be measured and the
schedule was built in advance. This thesis addresses the case those assumptions
exclude: not how to tune PID for one known plant, but how to re-tune it online
for a plant drawn from a hidden, unmeasured distribution.

## 2.2 Integral Windup and Anti-Windup Compensation

The textbook law of §2.1 assumes the commanded $u$ is realized. Real actuators
**saturate**: $u_{\text{sat}} = \mathrm{clip}(u, u_{\min}, u_{\max})$. When the
error keeps one sign during a long approach, the integrator accumulates

$$ I(T) = K_i \int_0^T e(\tau)\,d\tau \approx K_i\,\bar{e}\,T, $$

and the integral term alone can exceed the actuator range. Braking then
requires the derivative term to dominate, $K_d|\dot e| > I$; for the
parameters used in this thesis (a multi-metre approach at an actuator-limited
cruise speed) the accumulated integral exceeds the available derivative
authority by an order of magnitude, producing the large overshoot-and-recovery
transient analysed in Chapter 5. This phenomenon is **integral windup**.

*Worked example.* For the 5 m approach with $K_i = 0.7$, at the
actuator-limited cruise speed $v_{\max}\approx 0.5$ m/s the drive lasts
$T\approx 10$ s with mean error $\bar e\approx 2.5$ m, so
$I = K_i\,\bar e\,T \approx 0.7\times 2.5\times 10 = 17.5$ — already far beyond
the actuator command range $[-1,1]$. To arrest the vehicle the derivative term
must overcome this: $K_d|\dot e| > I$. With $K_d = 0.5$ and $|\dot e| = 0.5$
m/s the derivative contributes only $0.25$, short of $I$ by roughly
seventy-fold. The integrator must unwind — the vehicle overshoots, the error
changes sign, and only after seconds of reverse integration does the loop
recover. This matches the ~7 m overshoot and 74–93 s recovery measured for
naive PID in the no-reset environment (§5.6), and it is exactly what
anti-windup is designed to prevent.

Industrial PID blocks therefore include **anti-windup** compensation
[^astrom1995]. Two standard mechanisms are used here. **Back-calculation**
drives the integral state toward consistency with the saturated output through
a tracking term of time constant $T_t$,

$$ \dot{I} = K_i\,e + \tfrac{1}{T_t}\,(u_{\text{sat}} - u), $$

reducing to ordinary integration when unsaturated. **Conditional integration**
(clamping) suspends integration whenever the output is saturated and the error
would drive it further into saturation. Both are a few lines of code and ship
in every PLC and motor-drive firmware. The significance for this thesis is
methodological: comparing learned control against a PID *without* anti-windup
overstates the learner's advantage, whereas an environment aid that resets the
integrator using task knowledge (the `brake_integral_reset` of §3.x) *under*-
states it. The fair classical comparator sits between these extremes, and
Chapter 5 shows it solves the windup-prone task that naive PID fails.

## 2.3 Classical Adaptive Control

Adaptive control is the classical response to *changing* plant parameters, and
three families bracket the design space this thesis works in.

**Gain scheduling** stores gain sets indexed by a measured scheduling variable
and interpolates between them — the canonical example being an aircraft
autopilot scheduled on airspeed and altitude. It is effective and widely
deployed, but it presupposes two things this thesis denies: that the scheduling
variable is *measurable* at run time, and that the schedule was *designed
offline* from knowledge of how the dynamics vary with it. When the varying
parameter is hidden (no sensor) and the variation is not modelled in advance,
classical scheduling has nothing to index on. The learned policies here can be
read as gain scheduling where the schedule is *learned from experience* and
indexed on either the observed context (teacher) or an inferred latent
(student), removing both prerequisites.

**Self-tuning regulators** estimate the plant parameters online (e.g. by
recursive least squares on an ARX model) and re-solve for the controller gains
each step — certainty-equivalence adaptive control. They are powerful when the
plant admits a good linear-in-parameters model, but the online identification
is fragile under input saturation and unmodelled nonlinearity (exactly this
plant's regime), and they add an estimator whose convergence must itself be
guaranteed. The RL student is loosely analogous — it too infers something about
the plant online — but it learns the *whole* map from history to gains
end-to-end against the control objective, rather than chaining a separately
specified estimator and gain-solver.

**Model-reference adaptive control (MRAC)** adapts the controller online so the
closed loop tracks a reference model $y_m$. The MIT rule performs gradient
descent on the model-following error $e_m = y - y_m$,

$$ \dot{\theta} = -\gamma\, e_m \,\frac{\partial y}{\partial \theta}, $$

for controller parameters $\theta$ and adaptation gain $\gamma$. MRAC requires
a reference model whose behaviour the plant can actually realize and a
sensitivity $\partial y/\partial\theta$ with consistent sign; on nonlinear,
input-saturated plants both assumptions are fragile, and the MIT rule "may
become unstable" for large $\gamma$ or significant model mismatch
[^astrom1995adaptive]. The $\sigma$-modification of Ioannou and Tsakalis adds a
leakage term to bound parameter drift [^ioannou1986] but does not repair sign
inconsistency. Chapter 5 reports that MRAC, even with a *feasible* reference
model (one whose settling time the actuator-limited plant can achieve) and a
normalized MIT rule, fails entirely on this task — a negative result that
motivates the learning-based approach.

## 2.4 Reinforcement Learning, POMDPs, and Hidden-Parameter MDPs

### 2.4.1 MDPs and policy gradients

A Markov decision process is a tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma,
\rho_0)$ with states $\mathcal{S}$, actions $\mathcal{A}$, transition kernel
$P(s'|s,a)$, reward $R$, discount $\gamma\in[0,1)$, and initial distribution
$\rho_0$. A policy $\pi_\theta(a|s)$ induces a distribution over trajectories
$\tau=(s_0,a_0,s_1,\dots)$ and the objective $J(\theta) =
\mathbb{E}_{\tau\sim\pi_\theta}[\sum_t \gamma^t r_t]$. The state-value and
action-value functions are $V^\pi(s)=\mathbb{E}_\pi[\sum_{k\ge0}\gamma^k
r_{t+k}\mid s_t=s]$ and $Q^\pi(s,a)=\mathbb{E}_\pi[\sum_{k\ge0}\gamma^k
r_{t+k}\mid s_t=s,a_t=a]$, related by the Bellman expectation equation
$V^\pi(s)=\mathbb{E}_{a\sim\pi}[Q^\pi(s,a)]$ and
$Q^\pi(s,a)=R(s,a)+\gamma\,\mathbb{E}_{s'}[V^\pi(s')]$.

**Policy-gradient theorem.** Because only the trajectory *distribution* depends
on $\theta$, the gradient can be written without differentiating the (unknown)
dynamics. Using the log-derivative identity $\nabla_\theta p_\theta =
p_\theta\,\nabla_\theta\log p_\theta$ on the trajectory density
$p_\theta(\tau)=\rho_0(s_0)\prod_t \pi_\theta(a_t|s_t)P(s_{t+1}|s_t,a_t)$, the
transition and initial terms drop out of $\nabla_\theta\log p_\theta$ (they do
not depend on $\theta$), leaving

$$ \nabla_\theta J(\theta) = \mathbb{E}_{\tau\sim\pi_\theta}\!\Big[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)\,\Psi_t\Big], $$

where the weight $\Psi_t$ may be the return, the action-value $Q^\pi$, or — with
the lowest variance among unbiased choices — the **advantage**
$A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$. Subtracting the state-dependent baseline
$V^\pi(s)$ leaves the gradient unbiased (since
$\mathbb{E}_{a\sim\pi}[\nabla_\theta\log\pi_\theta(a|s)]=0$) while removing the
variance due to states being intrinsically good or bad rather than actions
being better or worse than average. This is the foundation on which §2.4.2–2.4.3
build: estimate $A_t$ well (GAE), and take the largest stable step toward
increasing $\mathbb{E}[\nabla\log\pi\cdot A]$ without invalidating the on-policy
samples (PPO's clip). That the dynamics never appear is exactly why the method
is **model-free** — the agent needs no model of how mass, friction, or actuator
strength map to motion, which is what makes it applicable to the HiP-MDP where
those parameters are hidden.

### 2.4.2 From trust regions to PPO

Vanilla policy-gradient ascent is unstable: a step that is too large in
parameter space can collapse the policy, and the on-policy data that justified
the gradient is invalidated by the very update it induces. Trust-region policy
optimization (TRPO) addresses this by maximizing a surrogate of the expected
return subject to a Kullback–Leibler constraint that keeps the new policy close
to the old. Writing the importance-sampling surrogate with the probability
ratio $r_t(\theta) = \pi_\theta(a_t|s_t)/\pi_{\theta_{\text{old}}}(a_t|s_t)$,
TRPO solves

$$ \max_\theta\ \mathbb{E}_t\!\big[ r_t(\theta)\,A_t \big] \quad \text{s.t.}\quad \mathbb{E}_t\!\big[ \mathrm{KL}\big(\pi_{\theta_{\text{old}}}(\cdot|s_t)\,\|\,\pi_\theta(\cdot|s_t)\big) \big] \le \delta. $$

The constrained problem requires a conjugate-gradient step on the Fisher
information matrix — accurate but expensive and awkward to implement. Proximal
Policy Optimization (PPO) [^schulman2017] replaces the hard KL constraint with a
*clipped* surrogate that removes the incentive to move the ratio outside
$[1-\epsilon, 1+\epsilon]$:

$$ L^{\text{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\big(r_t(\theta) A_t,\ \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\,A_t\big)\Big]. $$

The minimum makes the objective a pessimistic (lower) bound on the unclipped
surrogate: when $A_t > 0$ the clip caps the reward for increasing $r_t$ beyond
$1+\epsilon$, and when $A_t < 0$ it caps it for decreasing $r_t$ below
$1-\epsilon$; in both cases, once the ratio leaves the trust band the gradient
of that sample vanishes, so the update cannot be dominated by a few large ratio
moves. This recovers most of TRPO's stability with first-order optimization and
a few epochs of minibatch SGD per data batch. We use $\epsilon = 0.2$.

### 2.4.3 Advantage estimation (GAE)

The surrogate needs an advantage estimate $A_t$. The temporal-difference
residual $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is a one-step,
low-variance but biased estimate of the advantage; the full Monte-Carlo return
minus baseline is unbiased but high-variance. Generalized advantage estimation
(GAE) [^schulman2016] interpolates between them with an exponential weighting
parameter $\lambda$:

$$ A_t^{\text{GAE}(\gamma,\lambda)} = \sum_{l\ge 0}(\gamma\lambda)^l\,\delta_{t+l}. $$

At $\lambda=0$ this is the one-step TD residual (low variance, biased toward the
value function); at $\lambda=1$ it telescopes to the discounted Monte-Carlo
advantage $\sum_l \gamma^l r_{t+l} - V(s_t)$ (unbiased, high variance).
Intermediate $\lambda$ trades the two; we use $\gamma=0.99$, $\lambda=0.95$, the
standard continuous-control setting, which keeps most of the variance reduction
while admitting only mild bias. The critic $V_\phi$ is trained by regression to
the returns $A_t + V(s_t)$; the actor maximizes $L^{\text{CLIP}}$. The combined
loss adds a value term (coefficient 0.5) and an optional entropy bonus
(coefficient 0 here, since the Gaussian policy's learned log-std already
supplies exploration), and all linear layers use orthogonal initialization
[^saxe2014] with the standard $\sqrt 2$ gain on hidden layers and small gains on
the policy/value heads. Exact hyperparameters are in §3.4 and Appendix A.

### 2.4.4 POMDPs and hidden-parameter MDPs

When the agent cannot observe the full state it faces a **partially observable
MDP (POMDP)**: the optimal policy depends on the belief $b_t$ over hidden state
given the history $o_{1:t}, a_{1:t-1}$. The control problem here is a
**hidden-parameter MDP (HiP-MDP)** [^doshivelez2016]: a family of MDPs
$\mathcal{M}_\psi = (\mathcal{S}, \mathcal{A}, P_\psi, R, \gamma, \rho_0)$
indexed by a latent parameter $\psi = (m,\mu,\kappa)$ — mass, friction,
actuator strength — drawn per episode from $p(\psi)$ and held fixed within the
episode (§3.x adds mid-episode shifts). Only the transition kernel depends on
$\psi$. The objective averages over the parameter distribution,

$$ J(\theta) = \mathbb{E}_{\psi\sim p(\psi)}\,\mathbb{E}_{\tau\sim\pi_\theta, P_\psi}\Big[\sum_t \gamma^t r_t\Big]. $$

The **context-aware** agent observes $\psi$ and so faces an ordinary MDP in the
augmented observation. The **blind** agent does not observe $\psi$ and faces a
POMDP; it receives a stacked window of the last $k$ observations,
$\tilde{o}_t = (o_{t-k+1},\dots,o_t)$, as a finite-memory approximation of the
belief. Whether $k$ steps suffice to identify $\psi$ is an empirical question —
addressed by the stack-depth ablation (RQ3) and the probing analysis (RQ4).

### 2.4.5 Identifiability

Not every hidden parameter is identifiable from closed-loop data. For the
longitudinal car dynamics under wheel torque, mass enters only through
$\ddot{x} = F/m$ (observable during acceleration), actuator strength sets the
saturated cruise speed $v_{\max}\propto\kappa$ (observable during cruise), and
sliding friction does not bind while the wheels roll without slip (not
observable). Chapter 5 confirms this structure empirically through the probing
analysis, connecting the learning result to classical identifiability: the
information available to the blind agent is exactly the information its
closed-loop trajectory excites.

## 2.5 Domain Randomization and Teacher–Student Learning

**Domain randomization** trains across randomized simulator parameters so a
single policy generalizes over the distribution [^tobin2017][^peng2018].
Originally introduced to bridge the *visual* sim-to-real gap by randomizing
textures and lighting, it was quickly extended to *dynamics* randomization —
mass, friction, latency — for transferring control policies. A conceptual
subtlety, central to this thesis, is that domain randomization admits two
qualitatively different solutions. A memoryless policy can only become
**conservative-robust**: it learns a single behaviour that is adequate (if
suboptimal) across the whole parameter range, effectively marginalizing over
the unknown $\psi$. A policy with memory can instead become **adaptive**:
it infers $\psi$ (or a sufficient statistic of it) from the interaction history
and conditions its behaviour on that inference. The two are behaviourally hard
to distinguish from success rates alone — both can reach 100% — which is exactly
why this thesis adds a probing analysis (RQ4) that inspects the *representation*
to determine which has been learned, and a stack-depth ablation (RQ3) that
measures how much memory the adaptive solution actually requires.

**Context inference.** A line of work infers a latent context from history and
conditions the policy on it: RL² embeds the learning loop in a recurrent policy
[^duan2016]; PEARL learns a probabilistic context encoder [^rakelly2019];
UP-OSI regresses the physical parameters explicitly and feeds a parameter-
conditioned policy [^yu2017].

**Privileged teacher–student learning.** Rapid Motor Adaptation (RMA)
[^kumar2021] trains a teacher with privileged access to ground-truth
"extrinsics", then distils an adaptation module that regresses the teacher's
latent embedding from proprioceptive history — enabling deployment without
privileged sensors. This thesis adopts the same teacher/blind dichotomy with
two differences: the blind student is trained *from scratch* (not distilled),
isolating the observation regime under an otherwise identical pipeline; and the
action space is PID gain scheduling rather than raw torque, keeping the
classical baselines comparable and the learned policy interpretable.

**Frame stacking as memory.** Stacking $k$ past observations is the standard
finite-memory device for POMDP control; the thesis tests how much depth this
task actually needs and compares it against a learned recurrent memory (GRU).

**Relation to meta-reinforcement learning.** Because the agent "adapts to a new
task (parameter setting) at test time", the work is adjacent to meta-RL, and an
earlier framing of this project used that label. It is dropped here for
accuracy. Meta-RL proper — MAML-style methods that learn an initialization for
fast gradient adaptation, or RL² which learns an update rule embedded in a
recurrent policy — performs *learning* at test time, adjusting the policy from
test-time experience. The agents here do no test-time learning: the policy
weights are frozen at evaluation, and any adaptation is the forward pass of a
fixed policy conditioning on context (teacher) or on a finite history (student).
That is **amortized inference over a task distribution**, not meta-learning of
an adaptation procedure. The honest description — domain-randomized PPO with a
fixed observation window or recurrence — is preferred over the more fashionable
"meta-RL", and the distinction matters precisely because a strict reading of the
results (the flat stack ablation, the shallow gain schedule) shows the agents
are doing something closer to robust amortized control than to online
identification-and-retuning.

## 2.6 RL for PID Gain Scheduling, and the Gap

Prior RL-for-PID work splits into *offline tuning* — RL as an optimizer
returning one gain set per plant — and *online scheduling*, where gains vary
during operation. Online approaches typically assume either a known plant model
or direct measurement of the scheduling variable (the classical gain-scheduling
requirement). [CITATION PASS: insert 2–3 concrete online-RL-PID references with
their assumptions, replacing the two placeholders from the original §2.5; verify
the Shi et al. venue.] The gap this thesis fills: **online gain scheduling
where the scheduling variables (mass, friction, actuator strength) are hidden,
must be inferred from closed-loop behaviour, and can shift mid-episode** —
studied as a HiP-MDP, with a controlled teacher–student comparison, a
mechanistic probing analysis, a fair anti-windup baseline, and a cross-plant
transfer test.

[^astrom1995]: Åström, K. J. and Hägglund, T. (1995). *PID Controllers: Theory, Design, and Tuning*, 2nd ed. ISA.
[^astrom1995adaptive]: Åström, K. J. and Wittenmark, B. (1995). *Adaptive Control*, 2nd ed. Addison-Wesley.
[^ioannou1986]: Ioannou, P. and Tsakalis, K. (1986). A robust direct adaptive controller. *IEEE TAC* 31(11).
[^schulman2017]: Schulman, J. et al. (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347.
[^schulman2016]: Schulman, J. et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation. *ICLR*.
[^saxe2014]: Saxe, A., McClelland, J. and Ganguli, S. (2014). Exact solutions to the nonlinear dynamics of learning in deep linear networks. *ICLR*.
[^doshivelez2016]: Doshi-Velez, F. and Konidaris, G. (2016). Hidden Parameter Markov Decision Processes. *IJCAI*.
[^tobin2017]: Tobin, J. et al. (2017). Domain Randomization. *IROS*.
[^peng2018]: Peng, X. B. et al. (2018). Sim-to-Real Transfer with Dynamics Randomization. *ICRA*.
[^duan2016]: Duan, Y. et al. (2016). RL²: Fast RL via Slow RL. arXiv:1611.02779.
[^rakelly2019]: Rakelly, K. et al. (2019). Efficient Off-Policy Meta-RL via Probabilistic Context Variables (PEARL). *ICML*.
[^yu2017]: Yu, W. et al. (2017). Preparing for the Unknown: Learning a Universal Policy with Online System Identification. *RSS*.
[^kumar2021]: Kumar, A. et al. (2021). RMA: Rapid Motor Adaptation for Legged Robots. *RSS*.
