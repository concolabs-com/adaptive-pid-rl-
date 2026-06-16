# New Section (Ch. 2): Related Work — Context Inference and Privileged Learning

> Integration target: extends §2.4–2.5. Positions the thesis in the
> teacher–student / context-inference literature instead of the loose
> "meta-RL" framing. Citation list at the bottom needs verification pass.

## 2.z.1 Domain randomization

Domain randomization trains a policy across randomized simulator parameters so
that it generalises across the parameter distribution — originally for visual
sim-to-real transfer (Tobin et al., 2017) and subsequently for dynamics
parameters (Peng et al., 2018). The policy either becomes conservative-robust
(a single behaviour adequate everywhere) or, when given memory, *adaptive*
(behaviour conditioned on inferred parameters). Which of the two emerges is
precisely what the probing analysis in Chapter 5 measures.

## 2.z.2 Explicit context inference

A second line of work infers a latent context from interaction history and
conditions the policy on it: RL² embeds the learning loop in a recurrent
policy (Duan et al., 2016); PEARL learns a probabilistic context encoder over
trajectory snippets (Rakelly et al., 2019); system-identification approaches
regress the physical parameters explicitly and feed them to a
parameter-conditioned policy (Yu et al., 2017, "UP-OSI"). These methods make
the belief over hidden parameters an explicit object.

## 2.z.3 Privileged teacher–student learning

Rapid Motor Adaptation (RMA; Kumar et al., 2021) trains a *teacher* policy
with privileged access to the simulator's ground-truth extrinsics (mass,
friction, terrain), then distils an *adaptation module* that regresses the
teacher's latent extrinsics embedding from proprioceptive history, enabling
deployment without privileged sensors. Student-from-history designs of this
kind now dominate learned locomotion (e.g. Lee et al., 2020).

This thesis instantiates the same observation dichotomy — context-aware
teacher vs history-based blind student — with two deliberate differences.
First, the student is trained from scratch rather than distilled, so the
comparison isolates the *observation regime* under an otherwise identical
training pipeline (same architecture width, PPO objective, curriculum, data
budget). Second, the action space is PID gain scheduling rather than raw
torque: the learned component modulates an interpretable classical controller
instead of replacing it, which keeps the classical baselines (fixed PID,
anti-windup PID, MRAC) directly comparable in the same action space.

## 2.z.4 RL for PID tuning and gain scheduling

Prior work on RL for PID splits into offline tuning — RL as an optimiser that
returns one gain set per plant (e.g. entropy-maximising tuners, Bayesian-RL
hybrids) — and online scheduling, where gains vary during operation. Online
approaches frequently assume either a known plant model or direct measurement
of the scheduling variable (classical gain scheduling's requirement). The gap
addressed here: online gain scheduling where the scheduling variables (mass,
friction, actuator strength) are *hidden*, must be inferred from closed-loop
behaviour, and shift mid-episode. [CITATION PASS NEEDED: 2–3 concrete
online-RL-PID papers with their assumptions, replacing the two
CITATION_NEEDED placeholders from the old §2.5.]

## Citation list to verify

- Tobin, J. et al. (2017). Domain randomization for transferring deep neural
  networks from simulation to the real world. IROS 2017.
- Peng, X. B. et al. (2018). Sim-to-real transfer of robotic control with
  dynamics randomization. ICRA 2018.
- Duan, Y. et al. (2016). RL²: Fast reinforcement learning via slow
  reinforcement learning. arXiv:1611.02779.
- Rakelly, K. et al. (2019). Efficient off-policy meta-reinforcement learning
  via probabilistic context variables (PEARL). ICML 2019.
- Yu, W., Tan, J., Liu, C. K., Turk, G. (2017). Preparing for the unknown:
  Learning a universal policy with online system identification. RSS 2017.
- Kumar, A., Fu, Z., Pathak, D., Malik, J. (2021). RMA: Rapid Motor
  Adaptation for legged robots. RSS 2021.
- Lee, J. et al. (2020). Learning quadrupedal locomotion over challenging
  terrain. Science Robotics 5(47).
- Doshi-Velez, F., Konidaris, G. (2016). Hidden parameter Markov decision
  processes: A semiparametric regression approach for discovering latent task
  parametrizations. IJCAI 2016.
- Åström, K. J., Hägglund, T. (1995). PID Controllers: Theory, Design, and
  Tuning. 2nd ed., ISA.
- Åström, K. J., Wittenmark, B. (1995). Adaptive Control. 2nd ed.,
  Addison-Wesley.
- Ioannou, P., Tsakalis, K. (1986). A robust direct adaptive controller.
  IEEE Trans. Automatic Control 31(11).
- Schulman, J. et al. (2017). Proximal Policy Optimization algorithms.
  arXiv:1707.06347.
- Schulman, J. et al. (2016). High-dimensional continuous control using
  generalized advantage estimation. ICLR 2016.
- Todorov, E., Erez, T., Tassa, Y. (2012). MuJoCo: A physics engine for
  model-based control. IROS 2012.
- Saxe, A., McClelland, J., Ganguli, S. (2014). Exact solutions to the
  nonlinear dynamics of learning in deep linear networks. ICLR 2014.
- Haarnoja, T. et al. (2018). Soft Actor-Critic. ICML 2018.
