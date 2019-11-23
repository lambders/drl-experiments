"""
Implementation of Deep Q-Networks by the Google Brain team.
Reference:
    "Human-Level Control Through Deep Reinforcement Learning" by Mnih et al. 
"""

import os
import math
import random
import argparse
import numpy as np 
from collections import namedtuple

import torch
from tensorboardX import SummaryWriter

from game.wrapper import Game

# Global parameter which tells us if we have detected a CUDA capable device
CUDA_DEVICE = torch.cuda.is_available()



class DQN(torch.nn.Module):

    def __init__(self, options):
        """
        Initialize a Deep Q-Network instance.
        Uses the same parameters as specified in the paper.
        """
        super(DQN, self).__init__()

        self.opt = options
        
        self.conv1 = torch.nn.Conv2d(self.opt.len_agent_history, 32, 8, 4)
        self.relu1 = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(32, 64, 4, 2)
        self.relu2 = torch.nn.ReLU(inplace=True)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, 1)
        self.relu3 = torch.nn.ReLU(inplace=True)
        self.fc4 = torch.nn.Linear(3136, 512) 
        self.relu4 = torch.nn.ReLU(inplace=True)
        self.fc5 = torch.nn.Linear(512, self.opt.n_actions)


    def init_weights(self, m):
        """
        Initialize the weights of the network.

        Arguments:
            m (tensor): layer instance 
        """
        if type(m) == torch.nn.Conv2d or type(m) == torch.nn.Linear:
            torch.nn.init.uniform(m.weight, -0.01, 0.01)
            m.bias.data.fill_(0.01)


    def forward(self, x):
        """
        Forward pass to compute Q-values for given input states.

        Arguments:
            x (tensor): minibatch of input states

        Returns:
            tensor: state-action values of size (batch_size, n_actions)
        """
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = out.view(out.size()[0], -1)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)
        return out


Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory():

    def __init__(self, options):
        """
        Initialize a replay memory instance.
        Used by agent to create minibatches of experiences. Resuts in greater 
        data efficiency, reduced update variance, and smoother learning.
        """
        self.memory = []
        self.capacity = options.replay_memory_size


    def add(self, experience):
        """
        Add an experience to replay memory.

        Arguments:
            experience (Experience): add experience to replay memory 
        """
        self.memory.append(experience)

        # Remove oldest experience if replay memory full
        if len(self.memory) > self.capacity:
            self.memory.pop(0)


    def sample(self, batch_size):
        """
        Sample some transitions from replay memory.

        Arguments:
            batch_size (int): # of experiences to sample from replay memory

        Returns:
            dict: dictionary of random experiences if there are enough available, else None
        """
        if batch_size > len(self.memory):
            return None

        # Sample a batch
        sample = random.sample(self.memory, batch_size)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        sample = Experience(*zip(*sample))

        sample_batch = {
            'state': torch.stack(sample.state),
            'action': torch.tensor(sample.action).unsqueeze(1),
            'reward': torch.tensor(sample.reward),
            'next_state': torch.stack(sample.next_state),
            'done': torch.tensor(sample.done)
        }

        if CUDA_DEVICE:
            sample_batch['state'] = sample_batch['state'].cuda()
            sample_batch['action'] = sample_batch['action'].cuda()
            sample_batch['reward'] = sample_batch['reward'].cuda()
            sample_batch['next_state'] = sample_batch['next_state'].cuda()
            sample_batch['done'] = sample_batch['done'].cuda()

        return sample_batch



class DQNAgent:

    def __init__(self, options):
        """
        Initialize an agent instance.
        """
        self.opt = options

        # Replay memory buffer
        self.replay_memory = ReplayMemory(self.opt)

        # Epsilon used for selecting actions
        self.epsilon = np.linspace(
            self.opt.initial_exploration, 
            self.opt.final_exploration, 
            self.opt.final_exploration_frame
        )

        # Create network
        self.net = DQN(self.opt)
        if self.opt.mode == 'train':
            self.net.apply(self.net.init_weights)
            if self.opt.weights_dir:
                self.net.load_state_dict(torch.load(self.opt.weights_dir))
        if self.opt.mode == 'eval':
            self.net.load_state_dict(torch.load(self.opt.weights_dir))
            self.net.eval()

        if CUDA_DEVICE:
            self.net = self.net.cuda()

        # The optimizer
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.opt.learning_rate
        )

        # The flappy bird game instance
        self.game = Game(self.opt.frame_size) 

        # Log to tensorBoard
        if self.opt.mode == 'train':
            self.writer = SummaryWriter(self.opt.exp_name)

        # Loss
        self.loss = torch.nn.MSELoss()


    def select_action(self, state, step):
        """
        Use epsilon-greedy exploration to select the next action. 
        Controls exploration vs. exploitation in the network.

        Arguments:
            state (tensor): stack of four frames
            step (int): the current training step
            
        Returns:
            int: 0 if no flap, 1 if flap
        """
        # Make state have a batch size of 1
        state = state.unsqueeze(0)
        if CUDA_DEVICE:
            state = state.cuda()

        # Select epsilon
        step = min(step, self.opt.final_exploration_frame - 1)
        epsilon = self.epsilon[step]

        # Perform random action with probability self.epsilon. Otherwise, select
        # the action which yields the maximum reward.
        if random.random() <= epsilon:
            return np.random.choice(self.opt.n_actions, p=[0.95, 0.05])
        else:
            return torch.argmax(self.net(state)[0])


    def optimize_model(self):
        """
        Performs a single step of optimization.
        Samples a minibatch from replay memory and uses that to update the net.

        Returns:
            loss (float)
        """
        # Sample a batch [state, action, reward, next_state]
        batch = self.replay_memory.sample(self.opt.batch_size)
        if batch is None:
            return

        # Compute Q(s_t, a) 
        q_batch = torch.gather(self.net(batch['state']), 1, batch['action'])
        q_batch = q_batch.squeeze()

        # Compute V(s_{t+1}) for all next states
        q_batch_1, _ = torch.max(self.net(batch['next_state']), dim=1)
        y_batch = torch.tensor(
            [batch['reward'][i] if batch['done'][i] else 
            batch['reward'][i] + self.opt.discount_factor * q_batch_1[i] 
            for i in range(self.opt.batch_size)]
        )
        if CUDA_DEVICE:
            y_batch = y_batch.cuda()
        y_batch = y_batch.detach()

        # Compute loss
        loss = self.loss(q_batch, y_batch)

        # Optimize model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss


    def train(self):
        """
        Main training loop.
        """
        # Episode lengths
        eplen = 0

        # Initialize the environment and state (do nothing)
        frame, reward, done = self.game.step(0)
        state = torch.cat([frame for i in range(self.opt.len_agent_history)])

        # Start a training episode
        for i in range(1, self.opt.n_train_iterations):

            # Perform an action
            action = self.select_action(state, i)
            frame, reward, done = self.game.step(action)
            next_state = torch.cat([state[1:], frame])

            # Save experience to replay memory
            self.replay_memory.add(
                Experience(state, action, reward, next_state, done)
            )

            # Perform optimization
            loss = self.optimize_model()

            # Move on to the next state
            state = next_state

            # Save network
            if i % self.opt.save_frequency == 0:
                if not os.path.exists(self.opt.exp_name):
                    os.mkdir(self.opt.exp_name)
                torch.save(self.net.state_dict(), f'{self.opt.exp_name}/{str(i).zfill(7)}.pt')

            # Write results to log
            if i % self.opt.log_frequency == 0:
                self.writer.add_scalar('loss', loss, i)

            eplen += 1
            if done:
                self.writer.add_scalar('episode_length', eplen, i)
                eplen = 0


    def play_game(self):
        """
        Play Flappy Bird using the trained network.
        """

        # Initialize the environment and state (do nothing)
        frame, reward, done = self.game.step(0)
        state = torch.cat([frame for i in range(self.opt.len_agent_history)])

        # Start playing
        while True:

            # Perform an action
            state = state.unsqueeze(0)
            if CUDA_DEVICE:
                state = state.cuda()
            action = torch.argmax(self.net(state)[0])
            frame, reward, done = self.game.step(action)
            if CUDA_DEVICE:
                frame = frame.cuda()
            next_state = torch.cat([state[0][1:], frame])

            # Move on to the next state
            state = next_state

            # If we lost, exit
            if done:
                break

