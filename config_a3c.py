"""
The hyperparameters used for training the A3C Network. 
"""

# Number of actor-crtic workers
N_WORKERS = 8

# Refresh buffer after every x actions
BUFFER_UPDATE_FREQ = 20

# Number of frames in a state
AGENT_HISTORY_LENGTH = 4

# Optimizer parameter
LEARNING_RATE = 1e-6 

# Entropy regularization weight
ENTROPY_COEFF = 0.01

# Value loss coefficient 
VALUE_LOSS_COEFF = 0.5

# Discount factor weight
DISCOUNT = 0.99 

# Epsilon-greedy exploration parameters
INITIAL_EXPLORATION = 0.75
FINAL_EXPLORATION = 0.01
FINAL_EXPLORATION_FRAME = 1000000

# Number of output actions. In flappy bird, this is either do nothing or flap wings.
N_ACTIONS = 2

# Size of frame used to train the DQN
FRAME_SIZE = 84

# Interval in iterations at which to save network weights
SAVE_NETWORK_FREQ = 100000

# Number of iterations to train the network
TRAIN_ITERATIONS = 3000000

# Name that will be used for the result folder name
EXPERIMENT_NAME = 'exp_a3c'