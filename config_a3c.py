# Number of actor-learner threads
NUM_ACTOR_LEARNER_THREADS = 16

# Methods performed updates after every x actions
UPDATE_FREQ = 5

# Nujmber of frames until shared target network update
TARGET_NET_UPDATE_FREQ = 40000

# Number of times an action is repeated
ACTION_REPEAT = 4

# RMSProp 
RMSPROP_DISCOUNT = 0.99
RMSPROP_DECAY = 0.99

# Entropy regularization weight
ENTROPY_REG = 0.01

# Exploration
EXPLORATION_RATE = [0,4, 0.3, 0.3]
INITIAL_EXPLORATION = 1
FINAL_EXPLORATION = [0.1, 0.01, 0.5]
FINAL_EXPLORATION_FRAME = 4000000

N_EXPERIMENTS = 50
LEARNING_RATE = 
SAVE_NETWORK_FREQ = 100000