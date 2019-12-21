import argparse
from dqn import DQNAgent
from a2c import A2CAgent
from ppo import PPOAgent 


# ARGPARSER 
parser = argparse.ArgumentParser(description="drl-experiment options")
parser.add_argument("--algo",
                    type=str,
                    help="run the network in train or evaluation mode",
                    default="dqn",
                    choices=["dqn", "a2c", "ppo"])
parser.add_argument("--mode",
                    type=str,
                    help="run the network in train or evaluation mode",
                    default="train",
                    choices=["train", "eval"])

# DIRECTORY options
parser.add_argument("--exp_name",
                    type=str,
                    help="name of experiment, to be used as save_dir",
                    default="exp1")
parser.add_argument("--weights_dir",
                    type=str,
                    help="name of model to load",
                    default="")

# TRAIN options
parser.add_argument("--n_train_iterations",
                    type=int,
                    help="number of iterations to train network",
                    default=3000000) 
parser.add_argument("--learning_rate",
                    type=float,
                    help="learning rate",
                    default=1e-6) # DQN 1e-6, A2C 1e-4, PPO 1e-5
parser.add_argument("--len_agent_history",
                    type=int,
                    help="number of stacked frames to send as input to networks",
                    default=4)
parser.add_argument("--discount_factor",
                    type=float,
                    help="discount factor used for discounting return",
                    default=0.99)

# DQN specific options
parser.add_argument("--batch_size",
                    type=int,
                    help="batch size",
                    default=32)
parser.add_argument("--initial_exploration",
                    type=float,
                    help="epsilon greedy action selection parameter",
                    default=1.0)
parser.add_argument("--final_exploration",
                    type=float,
                    help="epsilon greedy action selection parameter",
                    default=0.01)
parser.add_argument("--final_exploration_frame",
                    type=int,
                    help="epsilon greedy action selection parameter",
                    default=1000000)
parser.add_argument("--replay_memory_size",
                    type=int,
                    help="maximum number of transitions in replay memory",
                    default=25000)

# A2C/PPO specific parameters
parser.add_argument("--n_workers",
                    type=int,
                    help="number of actor critic workers",
                    default=8)
parser.add_argument("--buffer_update_freq",
                    type=int,
                    help="refresh buffer after every x actions",
                    default=20)
parser.add_argument("--entropy_coeff",
                    type=float,
                    help="entropy regularization weight",
                    default=0.01)
parser.add_argument("--value_loss_coeff",
                    type=float,
                    help="value loss regularization weight",
                    default=0.5)
parser.add_argument("--max_grad_norm",
                    type=int,
                    help="norm bound for clipping gradients",
                    default=40)
parser.add_argument("--grad_clip",
                    type=float,
                    help="magnitude bound for clipping gradients",
                    default=0.1)

# LOGGING options
parser.add_argument("--log_frequency",
                    type=int,
                    help="number of batches between each tensorboard log",
                    default=100)
parser.add_argument("--save_frequency",
                    type=int,
                    help="number of batches between each model save",
                    default=100000)

# GAME options
parser.add_argument("--n_actions",
                    type=int,
                    help="number of game output actions",
                    default=2)
parser.add_argument("--frame_size",
                    type=str,
                    help="size of game frame in pixels",
                    default=84)



if __name__ == '__main__': 
    options = parser.parse_args()

    # Select agent
    if options.algo == 'dqn':
        agent = DQNAgent(options)
    elif options.algo == 'a2c':
        agent = A2CAgent(options)
    elif options.algo == 'ppo':
        agent = PPOAgent(options)
    else:
        print("ERROR. This algorithm has not been implemented yet.")

    # Train or evaluate agent
    if options.mode == 'train':
        agent.train()
    elif options.mode == 'eval':
        agent.play_game()
