# References

> Consolidated bibliography. All entries are real, well-known works; still
> **verify exact venue/volume/pages/year against the originals** before final
> submission (formatting depends on the citation style the supervisor sets —
> IEEE / APA / Harvard). Entries marked ⚠ are ones the author should confirm
> are the best representative for the claim they support (the niche online
> RL-for-PID literature in particular).

## Control theory — PID, tuning, anti-windup

1. Åström, K. J. and Hägglund, T. (1995). *PID Controllers: Theory, Design,
   and Tuning*, 2nd ed. Instrument Society of America. — PID formulation,
   tuning, anti-windup.
2. Åström, K. J. and Hägglund, T. (2006). *Advanced PID Control*. ISA. —
   anti-windup, set-point weighting.
3. Ziegler, J. G. and Nichols, N. B. (1942). Optimum settings for automatic
   controllers. *Transactions of the ASME*, 64, 759–768. — Z–N tuning.
4. Skogestad, S. (2003). Simple analytic rules for model reduction and PID
   controller tuning. *Journal of Process Control*, 13(4), 291–309. — SIMC.
5. Rivera, D. E., Morari, M. and Skogestad, S. (1986). Internal model control:
   PID controller design. *Ind. Eng. Chem. Process Des. Dev.*, 25(1), 252–265.
6. Bohn, C. and Atherton, D. P. (1995). An analysis package comparing PID
   anti-windup strategies. *IEEE Control Systems Magazine*, 15(2), 34–40. —
   back-calculation vs conditional integration.
7. Visioli, A. (2006). *Practical PID Control*. Springer. — windup, practical
   tuning.
8. O'Dwyer, A. (2009). *Handbook of PI and PID Controller Tuning Rules*, 3rd
   ed. Imperial College Press.

## Adaptive and robust control

9. Åström, K. J. and Wittenmark, B. (1995). *Adaptive Control*, 2nd ed.
   Addison-Wesley. — MRAC, MIT rule, self-tuning regulators.
10. Ioannou, P. A. and Sun, J. (1996). *Robust Adaptive Control*. Prentice
    Hall. — σ-modification, robustness of adaptive laws.
11. Ioannou, P. and Tsakalis, K. (1986). A robust direct adaptive controller.
    *IEEE Transactions on Automatic Control*, 31(11), 1033–1043.
12. Slotine, J.-J. E. and Li, W. (1991). *Applied Nonlinear Control*. Prentice
    Hall. — adaptive/nonlinear control background.
13. Rugh, W. J. and Shamma, J. S. (2000). Research on gain scheduling.
    *Automatica*, 36(10), 1401–1425. — gain-scheduling survey.
14. Ljung, L. (1999). *System Identification: Theory for the User*, 2nd ed.
    Prentice Hall. — persistent excitation, identifiability.

## Reinforcement learning — foundations

15. Sutton, R. S. and Barto, A. G. (2018). *Reinforcement Learning: An
    Introduction*, 2nd ed. MIT Press.
16. Bellman, R. (1957). *Dynamic Programming*. Princeton University Press.
17. Sutton, R. S., McAllester, D., Singh, S. and Mansour, Y. (2000). Policy
    gradient methods for RL with function approximation. *NeurIPS*. — policy-
    gradient theorem.
18. Williams, R. J. (1992). Simple statistical gradient-following algorithms
    for connectionist RL (REINFORCE). *Machine Learning*, 8, 229–256.
19. Kakade, S. (2002). A natural policy gradient. *NeurIPS*.

## Reinforcement learning — deep policy optimization

20. Schulman, J., Wolski, F., Dhariwal, P., Radford, A. and Klimov, O. (2017).
    Proximal Policy Optimization Algorithms. arXiv:1707.06347.
21. Schulman, J., Levine, S., Abbeel, P., Jordan, M. and Moritz, P. (2015).
    Trust Region Policy Optimization. *ICML*.
22. Schulman, J., Moritz, P., Levine, S., Jordan, M. and Abbeel, P. (2016).
    High-Dimensional Continuous Control Using Generalized Advantage
    Estimation. *ICLR*.
23. Mnih, V. et al. (2015). Human-level control through deep RL (DQN).
    *Nature*, 518, 529–533. — frame stacking precedent.
24. Mnih, V. et al. (2016). Asynchronous methods for deep RL (A3C). *ICML*.
25. Haarnoja, T., Zhou, A., Abbeel, P. and Levine, S. (2018). Soft
    Actor-Critic. *ICML*. — off-policy reference (SAC deferred here).
26. Lillicrap, T. P. et al. (2016). Continuous control with deep RL (DDPG).
    *ICLR*.
27. Engstrom, L. et al. (2020). Implementation matters in deep policy
    gradients: a case study on PPO and TRPO. *ICLR*. — PPO implementation
    details / orthogonal init.
28. Saxe, A., McClelland, J. and Ganguli, S. (2014). Exact solutions to the
    nonlinear dynamics of learning in deep linear networks. *ICLR*. —
    orthogonal initialization.

## Hidden parameters, domain randomization, meta-/context RL

29. Doshi-Velez, F. and Konidaris, G. (2016). Hidden Parameter Markov Decision
    Processes. *IJCAI*.
30. Tobin, J., Fong, R., Ray, A., Schneider, J., Zaremba, W. and Abbeel, P.
    (2017). Domain Randomization for Transferring Deep Neural Networks from
    Simulation to the Real World. *IROS*.
31. Peng, X. B., Andrychowicz, M., Zaremba, W. and Abbeel, P. (2018).
    Sim-to-Real Transfer of Robotic Control with Dynamics Randomization.
    *ICRA*.
32. Finn, C., Abbeel, P. and Levine, S. (2017). Model-Agnostic Meta-Learning
    (MAML). *ICML*.
33. Duan, Y. et al. (2016). RL²: Fast Reinforcement Learning via Slow
    Reinforcement Learning. arXiv:1611.02779.
34. Rakelly, K., Zhou, A., Finn, C., Levine, S. and Quillen, D. (2019).
    Efficient Off-Policy Meta-RL via Probabilistic Context Variables (PEARL).
    *ICML*.
35. Yu, W., Tan, J., Liu, C. K. and Turk, G. (2017). Preparing for the
    Unknown: Learning a Universal Policy with Online System Identification.
    *RSS*. — UP-OSI.
36. Kumar, A., Fu, Z., Pathak, D. and Malik, J. (2021). RMA: Rapid Motor
    Adaptation for Legged Robots. *RSS*.
37. Lee, J., Hwangbo, J., Wellhausen, L., Koltun, V. and Hutter, M. (2020).
    Learning Quadrupedal Locomotion over Challenging Terrain. *Science
    Robotics*, 5(47).
38. Akkaya, I. et al. (2019). Solving Rubik's Cube with a Robot Hand. arXiv:
    1910.07113. — automatic domain randomization.
39. Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory.
    *Neural Computation*, 9(8). — recurrent memory (GRU/LSTM lineage).
40. Cho, K. et al. (2014). Learning phrase representations using RNN
    encoder–decoder (GRU). *EMNLP*.

## Representation probing / interpretability

41. Alain, G. and Bengio, Y. (2017). Understanding intermediate layers using
    linear classifier probes. *ICLR Workshop*. — linear probing methodology.
42. Hewitt, J. and Liang, P. (2019). Designing and interpreting probes with
    control tasks. *EMNLP*. — probe validity / control tasks.

## RL for PID and learned control tuning (direct related work) — ⚠ verify

43. ⚠ Shi, Q., Lam, H.-K., Xiao, B. and Tsai, S.-H. (2020). Adaptive PID
    controller based on Q-learning. *Int. J. Systems Science* (confirm
    volume/pages).
44. ⚠ Carlucho, I., De Paula, M. and Acosta, G. G. (2020). An adaptive
    deep RL approach for MIMO PID control of mobile robots. *ISA
    Transactions* (confirm).
45. ⚠ Lawrence, N. P. et al. (2022). Deep reinforcement learning with shallow
    controllers: an experimental application to PID tuning. *Control
    Engineering Practice* (confirm).
46. ⚠ Younes Al Younes & Barczyk (2022) / Sedighizadeh & Rezazadeh — additional
    RL-PID candidates; the author should select the two or three most
    representative and verify details.

## Simulation / tooling

47. Todorov, E., Erez, T. and Tassa, Y. (2012). MuJoCo: A Physics Engine for
    Model-Based Control. *IROS*.
48. Brockman, G. et al. (2016). OpenAI Gym. arXiv:1606.01540.
49. Towers, M. et al. (2023). Gymnasium. (Farama Foundation; confirm citation
    form.)
50. Raffin, A. et al. (2021). Stable-Baselines3: Reliable RL implementations.
    *JMLR*, 22(268).
51. Paszke, A. et al. (2019). PyTorch: An imperative style, high-performance
    deep learning library. *NeurIPS*.

## Statistics / evaluation methodology

52. Efron, B. and Tibshirani, R. J. (1993). *An Introduction to the
    Bootstrap*. Chapman & Hall. — bootstrap CIs.
53. Wilson, E. B. (1927). Probable inference, the law of succession, and
    statistical inference. *JASA*, 22, 209–212. — Wilson score interval.
54. Welch, B. L. (1947). The generalization of "Student's" problem when
    several different population variances are involved. *Biometrika*, 34.
55. Cliff, N. (1993). Dominance statistics: ordinal analyses to answer ordinal
    questions. *Psychological Bulletin*, 114(3). — Cliff's delta.
56. Holm, S. (1979). A simple sequentially rejective multiple test procedure.
    *Scandinavian Journal of Statistics*, 6(2). — Holm–Bonferroni.
57. Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A. and Bellemare, M.
    (2021). Deep RL at the edge of the statistical precipice (rliable).
    *NeurIPS*. — RL evaluation methodology, IQM.
58. Henderson, P. et al. (2018). Deep RL that matters. *AAAI*. — seed
    variance, reproducibility in RL.
