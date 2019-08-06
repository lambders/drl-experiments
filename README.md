# hack-flappy-bird-drl
Training a DRL agent to play Flappy Bird. An exercise to reimplement DQN, Policy Gradient, and A3C DRL methods.

## Dependencies
Other package versions would probably work, I'm just listing my tested configuration.
- Python 3.6.4
- pyTorch
- pygame

## To Run

## Notes on DQN
I had difficulty getting the network to converge. Here are some things I did to help:
- For the optimizer, I used Adam instead of RMSProp that was used in the paper (inspired by nevenp's repo). 
- Also, to select random actions, I gave higher weight to the "do nothing" action since that action is used quite a bit more often than the "flap" action (inspired by xmfbit's repo).

## Results

## References
- DQN: https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning/
    - https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    - https://www.toptal.com/deep-learning/pytorch-reinforcement-learning-tutorial

- A3C: https://arxiv.org/pdf/1602.01783v1.pdf