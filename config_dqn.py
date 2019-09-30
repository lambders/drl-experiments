"""
The hyperparameters used for training the DQN. 
"""

# Number of training cases over which each SGD update is computed
MINIBATCH_SIZE = 32

# SGD updates are sampled from this number of most recent frames
REPLAY_MEMORY_SIZE = 25000

# Number of most recent frames experienced by the agent that are given as input to the Q network
AGENT_HISTORY_LENGTH = 4

# Discount factor using the Q-learning update (gamma)
DISCOUNT_FACTOR = 0.99

# Optimizer parameters
LEARNING_RATE = 1e-6 

# Epsilon-greedy exploration parameters
INITIAL_EXPLORATION = 1.0
FINAL_EXPLORATION = 0.01
FINAL_EXPLORATION_FRAME = 1000000

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

# The checkpoint of the network weights
# Will be the starting point for 'train' mode, will be frozen and used in 'eval' mode
# Put in an empty string id you wouldn't like a checkpoint state to be loaded
CHECKPOINT_DIR = 'easy/2900000.pt'

# The mode to run the network. Either train or eval.
MODE = 'eval'