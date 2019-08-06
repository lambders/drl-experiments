"""
The hyperparameters used for training the DQN. 
Most values were taken directly from the paper.
"""

# Number of training cases over which each SGD update is computed
MINIBATCH_SIZE = 32

# SGD updates are sampled from this number of most recent frames
REPLAY_MEMORY_SIZE = 10000

# Number of most recent frames experienced by the agent that are given as input to the Q network
AGENT_HISTORY_LENGTH = 4

# Number of paramater updates until the target network is updated
TARGET_NETWORK_UPDATE_FREQ = 10000

# Discount factor using the Q-learning update
DISCOUNT_FACTOR = 0.99

# Repeat each action selected by the agent this many times. 
# ACTION_REPEAT = 4

# Number of actions selected by the agent between successive SGD updates.
# UPDATE_FREQUENCY = 4

# RMSProp parameters
LEARNING_RATE = 1e-6 #0.00025
# GRADIENT_MOMENTUM = 0.95
# SQUARED_GRADIENT_MOMENTUM = 0.95
# MIN_SQUARED_GRADIENT = 0.01

# Epsilon-greedy exploration parameters
INITIAL_EXPLORATION = 0.1
FINAL_EXPLORATION = 0.0001
FINAL_EXPLORATION_FRAME = 1000000

# Uniform random policy is run for this number of frames before learning starts and the resulting experience is used to populate the replay memory
REPLAY_START_SIZE = 500000

# Maximum number of "do nothing" actions to be performed by the agent at the start of an episode
# NO_OP_MAX = 30

# Added by me!
# Number of output actions. In flappy bird, this is either do nothing or flap wings.
N_ACTIONS = 2

# 'cuda' or 'cpu'
DEVICE = 'cpu'

FRAME_SIZE = 84

SAVE_NETWORK_FREQ = 100000

TRAIN_ITERATIONS = 2000000