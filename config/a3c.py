"""
The hyperparameters used for training the A3C Network. 
"""

# Number of actor-crtic workers
N_WORKERS = 16

# Methods performed updates after every x actions
UPDATE_FREQ = 5

# Nujmber of frames until shared target network update
GLOBAL_NET_UPDATE_FREQ = 40000

# Number of times an action is repeated
AGENT_HISTORY_LENGTH = 4

# Optimizer parameter
LEARNING_RATE = 1e-6 

# Entropy regularization weight
ENTROPY_COEFF = 0.01

# Value loss coefficient 
VALUE_LOSS_COEFF = 0.5

# Discount factor weight
DISCOUNT = 0.99 ### TODO

# Number of output actions. In flappy bird, this is either do nothing or flap wings.
N_ACTIONS = 2

# Size of frame used to train the DQN
FRAME_SIZE = 84

# Interval in iterations at which to save network weights
SAVE_NETWORK_FREQ = 100000

# Numbe rof iterations to train the network
TRAIN_ITERATIONS = 3000000

# Name that will be used for the result folder name
EXPERIMENT_NAME = 'easy'

# 