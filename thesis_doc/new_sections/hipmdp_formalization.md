# New Section (Ch. 2): Problem Formalization — Hidden-Parameter MDPs

> Integration target: replaces/extends §2.3–2.4. Establishes the formal object
> the thesis studies and the exact distinction between the context-aware and
> blind agents. All equations verified against the implementation
> (`stage2_meta_rl_reproduction.py`, `envs/adaptive_suspension.py`).

## 2.x Markov Decision Processes

A Markov decision process (MDP) is a tuple $\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma, \rho_0)$
with state space $\mathcal{S}$, action space $\mathcal{A}$, transition kernel
$P(s_{t+1} \mid s_t, a_t)$, reward function $R(s_t, a_t, s_{t+1})$, discount
factor $\gamma \in [0, 1)$, and initial-state distribution $\rho_0$. A policy
$\pi_\theta(a \mid s)$ induces the objective

$$ J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t \, r_t \right], $$

where $\tau = (s_0, a_0, r_0, s_1, \dots)$ is a trajectory.

## 2.x+1 Hidden-Parameter MDPs

The control problem in this thesis is not a single MDP. The plant dynamics
depend on physical parameters — vehicle mass $m$, ground friction $\mu$, and
actuator strength $\kappa$ — that vary between deployments and are not part of
the standard sensor suite. Following Doshi-Velez and Konidaris (2016), this is
modelled as a **hidden-parameter MDP** (HiP-MDP): a family of MDPs

$$ \mathcal{M}_\psi = (\mathcal{S}, \mathcal{A}, P_\psi, R, \gamma, \rho_0), \qquad \psi = (m, \mu, \kappa) \sim p(\psi), $$

indexed by a latent parameter vector $\psi$ drawn once per episode and held
fixed within the episode (Section 3.x extends this to mid-episode parameter
shifts). Only the transition kernel depends on $\psi$; the reward function and
state/action spaces are shared.

The training distribution $p(\psi)$ is uniform over
$m \in [5, 20]\,\mathrm{kg}$, $\mu \in [0.1, 2.0]$, $\kappa \in [0.6, 1.4]$
(domain randomization). The optimisation objective becomes the expected return
over the parameter distribution:

$$ J(\theta) = \mathbb{E}_{\psi \sim p(\psi)} \, \mathbb{E}_{\tau \sim \pi_\theta, P_\psi} \left[ \sum_t \gamma^t r_t \right]. $$

Two observation regimes distinguish the agents studied:

**Context-aware (teacher) agent.** The observation includes the normalized
parameters: $o_t = (x_t, \dot{x}_t, e_t, a_{t-1}, \tilde\psi)$ with
$\tilde\psi = (m/m_0, \mu/\mu_0, \kappa)$. Given $\tilde\psi$, the process is
Markovian in $o_t$: the agent faces an ordinary MDP and can in principle
implement a direct mapping from parameters to gains — classical gain
scheduling with a learned schedule.

**Blind (student) agent.** The observation excludes $\tilde\psi$. The process
is then a **partially observable** MDP (POMDP): the optimal policy depends on
the belief $b_t(\psi) = p(\psi \mid o_{1:t}, a_{1:t-1})$, which must be
inferred from interaction history. The blind agent receives a stacked window
of the last $k$ observations,

$$ \tilde{o}_t = (o_{t-k+1}, \dots, o_t) \in \mathbb{R}^{6k}, $$

a finite-memory approximation of the belief state. Frame stacking turns the
POMDP into an approximate MDP on the window space; the approximation quality
depends on whether $k$ steps of history suffice to identify $\psi$ — an
empirical question addressed by the stack-size ablation (RQ3) and the probing
analysis (RQ4).

**Identifiability.** Not every parameter is identifiable from closed-loop
data. For this plant the longitudinal dynamics under wheel torque $\tau_w$,
wheel radius $r_w$ and viscous joint damping $b$ are approximately

$$ m\,\ddot{x} = \frac{2\,\kappa\,\tau(u)}{r_w} - \frac{2 b}{r_w^2}\,\dot{x}, $$

so the steady cruise speed $v_{\max} = \kappa\,\tau_{\max} r_w / (2b) \cdot 2/r_w^2$...
simplifying, $v_{\max} \propto \kappa$ is independent of $m$ and $\mu$: mass is
observable only in transients ($\ddot x = F/m$), actuator strength is
observable in both transients and cruise, and sliding friction does not enter
at all while the wheels roll without slip. Chapter 5 confirms this structure
empirically: probe decodability of $m$ peaks during acceleration and decays in
cruise, $\kappa$ remains decodable throughout, and $\mu$ is never decodable
above chance. This connects the learning-based analysis to classical
identifiability theory: the information available to the blind agent is
exactly the information excited by the closed-loop trajectory.

## 2.x+2 Teacher–student policies under privileged information

The context/blind pair instantiates the **privileged-information** paradigm:
a teacher trained with access to ground-truth parameters and a student that
must infer them from proprioceptive history. Rapid Motor Adaptation (RMA;
Kumar et al., 2021) trains the student by distilling the teacher's latent
embedding; RL² (Duan et al., 2016) and PEARL (Rakelly et al., 2019) learn
explicit context inference. The design here is deliberately simpler: the blind
agent is trained from scratch with the same PPO objective, isolating the
effect of the observation regime alone (identical architecture, reward,
curriculum, and data budget). The comparison therefore measures the *cost of
blindness* for this control task, not the benefit of a particular distillation
scheme.
