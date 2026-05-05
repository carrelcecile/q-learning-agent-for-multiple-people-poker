# Adaptive Q-learning for Multi-Player Texas Hold'em

A poker AI that learns to play four-player fixed-limit Texas Hold'em, then adapts its strategy in real time based on what kind of opponents it's facing.

## What this does

The project has three parts.

First, we built a poker game from scratch. Four players, simple betting rules (you can fold, check or call, or bet or raise), full hand scoring, blinds, and seat rotation. Nothing is borrowed from existing poker libraries.

Second, we trained a Q-learning agent to play. It learns by playing 20,000 hands against bots with different styles. We tried two training setups: one where the opponents stay the same throughout, and one where they switch every 5,000 hands.

Third, we added an adaptation layer. While playing, the agent watches its opponents and tries to guess their style (passive, aggressive, bluff-happy, tight, random, or mixed). Based on that guess, it tweaks its play. The clever part is that this happens without any retraining. The agent's underlying knowledge stays frozen, but its choices shift based on who it's up against.

## How to run it

Open the notebook in Colab or Jupyter and run all cells from top to bottom. It takes around 5 to 10 minutes on a regular laptop. No GPU needed.

```bash
pip install numpy pandas matplotlib scipy scikit-learn
jupyter notebook poker_revised.ipynb
```

## What's in the notebook

Section 1 builds the poker game itself. Section 2 creates five different bot opponents. Section 3 sets up the Q-learning agent. Section 4 trains it and compares it against the bots. Section 5 tests how well it copes with different opponent mixes. Section 6 trains a second agent on a changing opponent pool. Section 7 adds the adaptation layer and tests it.

## What we found

The Q-learning agent beats every rule-based bot in the setup it was trained on. We added confidence intervals to make sure the gap is real and not just luck.

The agent trained on changing opponents handles new situations more evenly than the one trained on a fixed setup, but it doesn't peak as high in any single scenario. This is a normal trade-off between being good at one thing and being okay at many things.

The adaptation layer helps most when opponents are passive or bluffing a lot. These styles are easy to spot from how often the bots fold, call, or raise. Against aggressive opponents, the layer doesn't help much, because aggressive and bluffing bots look very similar from the outside.

## Why we used Q-learning instead of CFR

Anyone who knows poker AI will ask this straight away. Counterfactual Regret Minimisation (CFR) is the algorithm behind Libratus and Pluribus, the systems that actually beat the best human players. If you want the strongest possible poker bot, CFR is the answer.

So why didn't we use it?

The first reason is that we were asking a different question. CFR finds a strategy that nobody can exploit. It plays the same way against a beginner as against a world champion. Our project is the opposite. We wanted to see if an agent could spot weaknesses in different opponents and take advantage of them. That kind of exploitation is what Q-learning is good at, and it's something CFR is designed to avoid.

The second reason is that CFR doesn't adapt while playing. It works out a strategy in advance, then sticks with it. There are research papers that combine CFR with online opponent modelling, but they take serious engineering. The whole point of our adaptation layer is that it works in real time with no extra computation. Adding CFR underneath would defeat the purpose.

The third reason is practical. Even simple versions of Hold'em have millions of game states for CFR to work through. Getting a CFR system to convergence takes hours or days on serious hardware. Our project runs in a 10-minute Colab notebook. Q-learning fits the scope. CFR doesn't.

## How CFR could be added later

If someone wanted to extend this work, the natural path would be to use CFR as the foundation rather than Q-learning. You'd train a CFR strategy as the base, then keep our adaptation layer on top. The base would handle solid equilibrium play, and the adaptation layer would handle exploitation when opponents leave gaps. The result would be stronger than either approach alone. It's a much bigger engineering job, but it's the obvious next step.

## What this project doesn't do

We're upfront about the limitations.

The betting rules are simplified. Real poker lets you bet any amount, which adds a whole layer of strategy that we don't model.

The hand strength estimator is a rough heuristic. A proper Monte Carlo equity calculator would be more accurate, but slower.

The bots don't learn while we train against them. In real competitive play, opponents adapt to you while you adapt to them. We don't model that.

The same player seat (Player 0) is always the learning agent. The dealer button rotates so positions vary, but the agent never plays from the other three seats.

The tabular approach won't scale. To handle richer information or more players, you'd need to switch to neural networks (a Deep Q-Network would be the natural next step).

## References

The full reference list is in the notebook and the report. Three of the most relevant ones:

Brown, N., and Sandholm, T. (2019). Superhuman AI for multiplayer poker. *Science*, 365(6456), 885 to 890.

Billings, D., Papp, D., Schaeffer, J., and Szafron, D. (1998). Opponent modeling in poker. *Proceedings of AAAI-98*, 493 to 499.

Zha, D. et al. (2021). RLCard: A platform for reinforcement learning in card games. *Proceedings of IJCAI-21*, 5264 to 5266.

## Authors

ST455 Group Project, LSE Department of Statistics. See the report for individual contributions.
