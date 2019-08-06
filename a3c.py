"""
Implementation of A3C by the Google DeepMind team.
Reference:
    "Asynchronous Methods for Deep Reinforcement Learning" by Mnih et al. 
"""

import os
import math
import random
import torch
import numpy as np 
from tensorboardX import SummaryWriter
from collections import namedtuple

import config_a3c as cfg
from game.wrapper import Game 


class ActorCriticNetwork(torch.nn.Module):

    def __init__(self):
        """
        Initialize an ActorCriticNetwork instance.
        Uses the same parameters as specified in the paper.
        """
        super(ActorCriticNetwork, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(cfg.AGENT_HISTORY_LENGTH, 16, 8, 4)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(16, 32, 4, 2)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(3136, 256) # TODO: Don't hard code
        self.relu3 = torch.nn.ReLU()
        self.actor = torch.nn.Linear(256, cfg.N_ACTIONS)
        self.crtic = torch.nn.Linear(256, 1)
        self.softmax1 = torch.nn.SoftMax()
        # The model used by actor-critic agents had two set of outputs – a softmax output with one entry per action representing the probability of selecting the action, and a single linear output representing the value function


    def forward(self, x):
        """
        Forward pass to compute Q-values for given input states.

        Arguments:
            x (tensor): minibatch of input states

        Returns:
            tensor: state-action values of size (batch_size, n_actions)
            tensor: value function of size (batch_size, 1)
        """
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = x.view(x.size()[0], -1)
        x = self.relu3(self.fc3(x))
        probability = self.softmax1(self.actor(x))
        value = self.critic(x)
        return probability, value



class ActorCriticThread(mp.Process):

    def __init__(self):
        """
        Initialize an actor thread subprocess. 
        A bit of a misnoomer in this case since we are using multiprocessing 
        instead of multithreading. The reasoning behind this is that pyTorch
        has better support for multiprocessing (i.e., built-in library).
        """
        # Epsilon used for selecting actions
        self.epsilon = np.logspace(
            math.log(cfg.INITIAL_EXPLORATION), 
            math.log(cfg.FINAL_EXPLORATION), 
            num=cfg.FINAL_EXPLORATION_FRAME, 
            base=math.e
        )

        # Create local ACNetwork
        self.net = ActorCriticNetwork().to(cfg.DEVICE)

        # The optimizer
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr = cfg.LEARNING_RATE
        )

        self.device = cfg.DEVICE

        # The flappy bird game instance
        self.game = Game(cfg.FRAME_SIZE)

        # Log to tensorBoard
        self.writer = SummaryWriter('log')

        # Loss
        self.loss = torch.nn.MSELoss()

        # Buffer
        self.buffer = {'state': [], 'action': [], 'reward': [], 'value': []}


    def select_action(self, state, step):
        """
        Use epsilon-greedy exploration to select the next action. 
        Controls exploration vs. exploitatioin in the network.

        Arguments:
            state (tensor): 
            step (int): 
        Returns:
            int: 
        """
        # Make state have a batch size of 1
        state = state.unsqueeze(0)
        # Select epsilon
        epsilon = self.epsilon[min(step, cfg.FINAL_EXPLORATION_FRAME-1)]

        # Perform random action with probability self.epsilon. Otherwise, select the action which yields the maximum reward.
        if random.random() < epsilon:
            return np.random.choice(cfg.N_ACTIONS, p=[0.9, 0.1])
        else:
            with torch.no_grad():
                return torch.argmax(self.net(state), dim=1)[0]


    def optimize_model(self, next_state, done):
        """
        Performs a single step of optimization.
        Samples a minibatch from replay memory and uses that to update the net.


        Returns:
            loss (float)
        """
        # Calculate the value of the next state
        if done:
            value_1 = 0
        else:
            value_1 = self.net(next_state)[1]

        # Calculate reward
        for reward in self.reward[::-1]:
            value_1 = reward + gamma * value_1
            self.buffer['value'].append(value_1)
        self.buffer['value'].reverse()

        # Calculate loss
        loss = self.net.loss(self.buffer)

        # Calculate gradients
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        # Reset the buffers
        self.buffer = dict.fromkeys(self.buffer, [])



    def train(self):
        """
        """
        # Episode lengths
        episode_length = []
        eplen = 0

        # Initialize the environment and state (do nothing)
        frame, reward, done = self.game.step(0)
        state = torch.cat([frame for i in range(cfg.AGENT_HISTORY_LENGTH)]).to(cfg.DEVICE)


        # Start a training episode
        for i in range(1, cfg.TRAIN_ITERATIONS):

            # Perform an action
            action = self.select_action(state.to(cfg.DEVICE), i)
            frame, reward, done = self.game.step(action)
            next_state = torch.cat([state[1:], frame.to(cfg.DEVICE)])

            # Save experience to buffer
            self.buffer['state'].append(state)
            self.buffer['action'].append(action)
            self.buffer['reward'].append(reward)

            # Perform optimization
            # Sample random minibatch and update policy network
            loss = self.optimize_model(done)

            # Synchronize networks
            if i % cfg.TARGET_NET_UPDATE_FREQ == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # Save network
            if i % cfg.SAVE_NET_FREQ == 0:
                if not os.path.exists('results'):
                    os.mkdir('results')
                torch.save(self.target_net.state_dict(), f'results/{str(i).zfill(7)}.pt')

            # Write results to log
            if i % 100 == 0:
                self.writer.add_scalar('loss', loss, i)

            eplen += 1
            if done:
                print(i, eplen)
                self.writer.add_scalar('episode_length', eplen, i)
                eplen = 0

            # Move on to next state
            state = next_state


    def play_game():
        return None



def Trainer():
    def __init__(self):
        # The shared network for all local processes
        # Make sure the shared network's parameters are available in multiprocessing
        self.net = ActorCriticNetwork()
        self.net.share_memory()

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr = cfg.LEARNING_RATE
        )
        self.workers = [ActorCriticWorker() for i in range(cfg.NUM_ACTOR_LEARNER_THREADS)] 

    def train(self):
        for worker in self.workers:
            worker.start()

        while True:
            r = results_queue.get()
            if r is not None:
                res.append(r)
            else:
                break
        [w.join() for w in workers]

            worker.net.load_state_dict(self.target_net.state_dict())
            worker.train()