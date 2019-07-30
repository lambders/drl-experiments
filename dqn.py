"""
Implementation of Deep Q-Networks by the Google Brain team.

Implementor: Amanda Vu

Reference:
    "Human-Level Control Through Deep Reinforcement Learning" by Mnih et al. 
"""

import math
import random
import torch
import numpy as np 
import tensorboardX

import config as cfg
from game.main import Game 

#TODO Plot episode durations
class DQN(nn.Module):

    def __init__(self):
        """
        Initialize a Deep Q-Network instance.
        """
        # Game screenshot dimensions
        self.h = cfg.FRAME_SIZE
        self.w = cfg.FRAME_SIZE

        # The number of "channels" of the input. In DQN, the # channels refers to the agent history length. The number of most recent frames experienced by the agent that are given as input to the Q network.
        self.n_channels = cfg.AGENT_HISTORY_LENGTH

        # Number of output actions. In flappy bird, this is either do nothing or flap wings.
        self.n_actions = cfg.N_ACTIONS
        

        # Build the network. Used the same parameters as specified in the paper.
        self.conv1 = torch.nn.Conv2d(self.n_channels, 32, 8, 4)
        self.relu1 = nn.ReLU()
        self.conv2 = torch.nn.Conv2d(32, 64, 4, 2)
        self.relu2 = nn.ReLU()
        self.conv3 = torch.nn.Conv2d(64, 64, 3, 1)
        self.relu3 = nn.ReLU()
        self.fc4 = torch.nn.Linear(3136, 512) # TODO: Don't hard code
        self.relu4 = nn.ReLU()
        self.fc5 = torch.nn.Linear(512, self.n_actions)


    def forward(self, x):
        """
        Forward pass of the network to compute the Q-value for some given input states.

        Arguments:
            x (tensor): minibatch of input states

        Returns:
            tensor: Q-values for every possible action taken in the input states x
        """
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = x.view(x.size()[0], -1)
        x = self.relu4(self.fc4(x))
        x = self.fc5(x)
        return x


class ReplayMemory():

    def __init__(self):
        """
        Initialize a replay memory instance.
        This allows the agent to apply Q-learning updates over a sampling ot its experiences. This allows for several unique advantages:
            - Greater data efficiency since each step of experience is potentially used in many weight updates.
            - Reduces variance of the DQN updates since randomizing the samples breaks time correlations between consecutive samples.
            - Smooths out learning and avoids oscillation or divergence in network parameters since the behavior distribution is averaged over many of its previous states.

        """
        self.memory = []
        self.capacity = cfg.REPLAY_MEMORY_SIZE


    def add(self, experience):
        """
        Add an experience to replay memory.

        Arguments:
            experience (list): The [state, action, reward, next_state, done] transition of the most recent step
        """
        # Add the experience to replay memory
        self.memory.append(experience)

        # Remove oldest experience if replay memory full
        if len(self.memory) > self.capacity:
            replay_memory.pop(0)


    def sample(self, batch_size):
        """
        Sample some transitions from replay memory.

        Arguments:
            batch_size (int): the number of experiences to sample from replay memory

        Returns:
            list [Experience]: list of random experiences if there are enough available, else None
        """
        if batch_size > len(self.memory):
            return None

        sample = random.sample(self.memory, batch_size)
        return {
            'state': sample[:,0],
            'action': sample[:,1],
            'reward': sample[:,2],
            'next_state': sample[:,3],
            'done': sample[:,4]
        }


class Agent:

    def __init__(self):
        """
        Initialize an agent instance.
        The agent will learn an optimal policy that will allow it to play the game.
        """
        self.replay_memory = ReplayMemory()
        self.epsilon = np.logspact(math.log(INITIAL_EXPLORATION), math.log(FINAL_EXPLORATION), num=FINAL_EXPLORATION_FRAME, base=math.e)

        self.target_net = DQN()
        self.policy_net = DQN()
        self.target_net.load_state_dict(policy_net.state_dict())
        self/target_net.eval()

        # The optimizer
        self.optimizer = torch.optim.RMSprop(
            params = policy_net.parameters(),
            lr = cfg.LEARNING_RATE,
            momentum = cfg.GRADIENT_MOMENTUM,
            alpha = cfg.SQUARED_GRADIENT_MOMENTUM,
            eps = cfg.MIN_SQUARED_GRADIENT
        )
        # self.optimizer = torch.optim.Adam(
        #     policy_net.parameters(),
        #     lr = cfg.LEARNING_RATE
        # )

        self.device = cfg.DEVICE

        self.game = Game()

        # Number of parameter updates so far
        self.steps = 0

        # Log
        self.writer = SummaryWriter('log')

        # Loss
        self.loss = torch.nn.MSELoss()


    def select_action(self, state):
        """
        Use epsilon-greedy exploration to select the next action. Controls exploration vs. exploitatioin in the network.

        Arguments:
            action_space (tensor): 
        Returns:
            tensor: 
        """
        # Select epsilon
        if self.step > cfg.FINAL_EXPLORATION_FRAME:
            epsilon = self.epsilon[-1]
        else:
            epsilon = self.epsilon[self.step]

        # Perform random action with probability self.epsilon. Otherwise, select the action which yields the maximum reward.
        if random.random() < epsilon:
            return torch.tensor([random.randrange(cfg.N_ACTIONS)])
        else:
            with torch.no_grad():
                return torch.argmax(self.policy_net(state))


    def optimize_model(self):
        """
        Performs a single step of optimization.
        Samples a minibatch from replay memory and uses that to update the policy_net.
        """
        # Sample a batch [state, action, reward, next_state]
        batch = self.replay_memory.sample(cfg.MINIBATCH_SIZE)
        if batch is None:
            return

        # Compute mask of non-final states

        # Use policy_net to find best action
        best_action = torch.argmax(self.policy_net(batch['state']))
        # Use target_net to get y_batch
        q_batch_expected = 
        # Use policy_net on best action to get q_batch

        # Compute Q-value using policy net, Q(s_t, a)
        q_batch = torch.sum(self.policy_net(batch['state']) * batch['action'])

        # y_batch = expected q_batch
        # Compute expected Q-value y using target net. Using a  network with an older set of parameters adds a delay between the time an update to the network is made and the time the update affects targets y, stabilizing training
        q_batch_old = self.target_net(batch['state'])
        y_batch = batch['reward'] + cfg.DISCOUNT_FACTOR * q_batch_old

        # Clip the reward to be between -1 and 1 for further training stability 

        # Compute loss
        loss = self.loss(q_batch, y_batch)
        self.writer.add_scalar('loss', loss, self.step)

        # Optimize model
        self.optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()


    def train(self):
        """
        """
        # Initialize the environment and state (do nothing)
        frame, reward, done = self.game.step(0)
        state = torch.cat([frame for i in range(cfg.AGENT_HISTORY_LENGTH)])


        # Start a training episode
        for i in range(cfg.TRAIN_ITERATIONS):
            # Update step counter
            self.steps = i

            # Perform an action
            action = self.select_action(state)
            frame, reward, done = self.game.step(action)
            next_state = torch.cat(state[1:,:,:], frame)

            # Save experience to replay memory
            self.replay_memory.add([state, action, reward, next_state, done])

            # Perform optimization
            # Sample random minibatch and update policy network
            self.optimize()

            # Move on to the next state
            state = next_state

            # Update the target network
            if self.steps % cfg.TARGET_NETWORK_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # Save network
            if self.steps % SAVE_NETWORK_FREQ == 0:
                save_checkpoint(target_net.state_dict())


    def play_game():
        return None

