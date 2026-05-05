# Adaptive Q-learning for Multi-Player Texas Hold'em

A tabular Q-learning agent for simplified four-player fixed-limit Texas Hold'em, with an inference-time adaptation layer that adjusts policy based on detected opponent style. Built as the ST455 group project at LSE.

---

## What this does

Three things, in order:

1. **A poker environment.** Four players, fixed-limit betting, three discrete actions (fold / check-call / bet-raise), full hand evaluation, blinds, position rotation. Self-contained — no external poker library.

2. **A Q-learning agent.** Tabular, with a ten-dimensional discrete state encoding (round, hand strength, draw strength, pot size, stack size, position, betting pressure, etc.). Trained for 20,000 hands against a fixed mix of passive, aggressive, and bluff-heavy opponents, then against a time-varying pool that rotates every 5,000 hands.

3. **An adaptive policy layer.** A rolling 100-action classifier identifies the current opponent style (passive / aggressive / bluff-heavy / tight / random / mixed). The agent's frozen Q-values are then nudged at inference time based on the detected style — without any re-training. This is the project's main methodological contribution.

---

## Quick start

Open the notebook in Colab or Jupyter and run all cells top to bottom. Total runtime ≈ 5–10 minutes on CPU.

```bash
pip install numpy pandas matplotlib scipy scikit-learn
jupyter notebook poker_revised.ipynb
```

No GPU required. No external RL framework required.

---

## Project structure

| Notebook section | What it does |
|---|---|
| 1 | Poker environment — cards, players, hand evaluation, full game loop |
| 2 | Five rule-based opponents (Random, Passive, Aggressive, BluffHeavy, Tight) |
| 3 | State encoding + tabular Q-learning agent |
| 4 | Train against fixed pool; compare vs. baselines with 95% CIs |
| 5 | Evaluate trained agent across five different opponent compositions |
| 6 | Train second agent on time-varying pool; compare generalisation |
| 7 | Adaptive layer + opponent tracker + classifier confusion matrix |

---

## Key results

- **Q-learning beats every rule-based baseline** in the training distribution, with confidence intervals confirming the gap is statistically meaningful (not just noise).
- **The time-varying agent generalises more evenly** across opponent pools than the fixed-pool agent, but at a small cost in peak performance — the standard breadth-vs-depth trade-off.
- **The adaptive layer helps most against passive and bluff-heavy pools**, where the classifier signal is cleanest. Against aggressive pools it performs comparably to the fixed policy, because aggressive and bluff-heavy agents are hard to distinguish from action frequencies alone.

The classifier confusion matrix in Section 7.4 quantifies exactly where the adaptation succeeds and where it fails.

---

## Why Q-learning, not CFR?

Anyone familiar with the poker AI literature will immediately ask: why not Counterfactual Regret Minimisation? CFR is the algorithm behind Libratus and Pluribus, the only systems to achieve superhuman performance in heads-up and six-player no-limit Texas Hold'em respectively. It is, by a substantial margin, the state of the art for this domain.

The honest answer has two parts.

### What CFR does well

CFR is purpose-built for imperfect-information extensive-form games. It iteratively traverses the game tree, computes counterfactual regrets at each information set, and provably converges to a Nash equilibrium in two-player zero-sum games. For a problem as large and structured as Hold'em, CFR with abstraction (and its modern variants — CFR+, MCCFR, Deep CFR) is unambiguously the better algorithmic choice if your goal is **near-optimal play against a sophisticated opponent**.

### Why we chose not to use it

Three reasons, in order of importance.

**1. Our research question is about adaptation, not optimality.** A Nash equilibrium strategy is by construction unexploitable — but it also deliberately ignores opponent tendencies. Against a calling station who never folds, a Nash agent does not start value-betting more; it plays the same balanced strategy it would against a world-class opponent. Our project asks the opposite question: can an agent identify opponent style and exploit it? Q-learning, with its capacity to develop opponent-specific value estimates, is the appropriate tool for that question. CFR would actively work against the hypothesis we wanted to test.

**2. CFR does not adapt online.** Standard CFR computes a fixed strategy from offline tree traversal. Variants exist for online adaptation (e.g. opponent modelling layered on top of CFR equilibria), but these are research-grade systems requiring substantial engineering. The whole point of our adaptive layer is that it operates in real time with no re-computation. Bolting CFR onto that architecture would defeat the purpose.

**3. Computational scope.** Even simplified Hold'em variants generate game trees with millions of information sets. CFR with meaningful card abstraction takes hours to days on dedicated hardware for converged play. Our project budget is a 10-minute Colab notebook. Q-learning with a 10,000-state Q-table runs in minutes and produces results the marker can actually inspect end-to-end.

### How CFR could be added

For anyone extending this work, the natural integration would be:

- **Train a CFR base policy** for the four-player game on a coarse abstraction (suit-isomorphism, bucketed bet sizing). Tools like OpenSpiel provide reference implementations.
- **Replace the frozen Q-table** in the AdaptiveQLearningAgent with the CFR strategy as the equilibrium baseline.
- **Keep the OpponentBehaviorTracker and adjustment layer.** The adaptive logic — "passive pool → bet more for value" — is independent of how the base strategy was computed. The contribution becomes: a Nash baseline + interpretable exploitation layer.

This would be a strictly better system. It is also a substantially harder engineering project, well beyond the scope of a six-week coursework deadline. We would frame it as the natural next step rather than a missed opportunity.

---

## Limitations

The notebook is honest about what it doesn't do.

- **Fixed-limit betting only.** No-limit poker, where bet sizing is itself a strategic decision, is qualitatively harder and not modelled here.
- **Coarse hand strength heuristic.** A real Monte Carlo equity calculator would dominate ours, at the cost of runtime.
- **Static rule-based opponents.** Our opponents do not adapt to the agent during training — the genuine adversarial co-adaptation problem is not addressed.
- **Single RL seat.** Player 0 is always the agent. Positional generalisation is partial because the dealer button rotates, but the agent never plays from seats 1–3.
- **Tabular method scalability.** This approach does not extend to richer state representations or larger games without function approximation. The natural extension is a small DQN.

---

## References

The full reference list is in the notebook (Section 7.5) and the report. Three particularly relevant items:

- Brown, N., & Sandholm, T. (2019). Superhuman AI for multiplayer poker. *Science*, 365(6456), 885–890.
- Billings, D., Papp, D., Schaeffer, J., & Szafron, D. (1998). Opponent modeling in poker. *Proceedings of AAAI-98*, 493–499.
- Zha, D. et al. (2021). RLCard: A platform for reinforcement learning in card games. *Proceedings of IJCAI-21*, 5264–5266.

---

## Authors

ST455 Group Project, LSE Department of Statistics. See the report for individual contributions.
