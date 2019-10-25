"""
Implementation of A3C by the Google DeepMind team.
Reference:
    "Asynchronous Methods for Deep Reinforcement Learning" by Mnih et al. 
"""

import os
import math
import random
import torch
from torch.distributions.categorical import Categorical
import numpy as np 
from tensorboardX import SummaryWriter
from collections import namedtuple

import config_a3c as cfg
from game.wrapper import Game 

# TODO: Log

class ActorCriticNetwork(torch.nn.Module):

    def __init__(self):
        """
        Initialize an ActorCriticNetwork instance. The actor has an output for 
        each action and the critic provides the value output
        Uses the same parameters as specified in the paper.
        """
        super(ActorCriticNetwork, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(cfg.AGENT_HISTORY_LENGTH, 16, 8, 4)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(16, 32, 4, 2)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(2592, 256) # TODO: Don't hard code
        self.relu3 = torch.nn.ReLU()
        self.actor = torch.nn.Linear(256, cfg.N_ACTIONS)
        self.critic = torch.nn.Linear(256, 1)
        self.softmax = torch.nn.Softmax()
        self.logsoftmax = torch.nn.LogSoftmax()


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
            int: selected action, 0 to do nothing and 1 to flap
            float: entropy of action space
            float: log probability of selecting the action 
            float: value of the particular state
        """
        # Forward pass
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = x.view(x.size()[0], -1)
        x = self.relu3(self.fc3(x))
        action_logits = self.actor(x)
        value = self.critic(x)
        return value, action_logits


    def act(self, x):
        """
        Returns:
            tensor(8,1)
        """
        # Forward pass
        values, action_logits = self.forward(x)
        probs = self.softmax(action_logits)
        log_probs = self.logsoftmax(action_logits)

        # Choose action stochastically
        actions = probs.multinomial(1)

        # Evaluate action
        action_log_probs = log_probs.gather(1, actions)
        dist_entropy = -(log_probs * probs).sum(-1).mean()
        return values, actions, action_log_probs

    def evaluate_actions(self, x, actions):
        # Forward pass 
        value, action_logits = self.forward(x)
        probs = self.softmax(action_logits)
        log_probs = self.logsoftmax(action_logits)

        # Evaluate actions
        action_log_probs = log_probs.gather(1, actions)
        dist_entropy = -(log_probs * probs).sum(-1).mean()
        return value, action_log_probs, dist_entropy


Experience = namedtuple('Experience', ('state', 'action', 'action_log_prob', 'value', 'reward', 'mask'))

class Agent():

    def __init__(self):
        """
        Initialize an A2C Instance. 
        """
        # Create ACNetwork
        self.net = ActorCriticNetwork()
        self.net.apply(self.net.init_weights)

        # Optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.LEARNING_RATE)

        # The flappy bird game instance
        self.games = [Game(cfg.FRAME_SIZE) for i in range(cfg.N_WORKERS)]

        # Log to tensorBoard
        self.writer = SummaryWriter(cfg.EXPERIMENT_NAME)

        # Buffer
        self.memory = []


    def optimize_model(self, next_state, done):
        """
        Performs a single step of optimization.

        Arguments:
            next_state (tensor): next frame of the game
            done (bool): True if next_state is a terminal state, else False

        Returns:
            loss (float)
        """
        # Calculate the value of the next state
        next_value, _ = self.net(self.memory['state'][-1])

        # Compute returns
        returns = self.memory['returns']
        returns[-1] = next_value
        for i in reversed(range(cfg.BUFFER_UPDATE_FREQ)):
            returns[i] = returns[i+1] * cfg.DISCOUNT * self.memory['masks'][i+1] + rewards[i]

        # Evaluate actions
        values, action_log_probs, dist_entropy = actor_critic.evaluate_actions(self.memory['states'][:-1], self.memory['actions'])
        values = values.view(cfg.BUFFER_UPDATE_FREQ, cfg.N_WORKERS, 1)
        action_log_probs = action_log_probs.view(cfg.BUFFER_UPDATE_FREQ, cfg.N_WORKERS, 1)

        # Compute losses
        advantages = returns[:-1] - values
        value_loss = advantages.pow(2).mean()
        action_loss = -advantages * action_log_probs.mean()
        loss = value_loss * cfg.VALUE_LOSS_COEFF + action_loss - dist_entropy * cfg.ENTROPY_COEFF

        # Optimizer step
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(self.net.parameters(), cfg.MAX_GRAD_NORM)
        self.opt.step()

        return loss


    def env_step(self, states, actions):
        next_state_list, reward_list, done_list = [], [], []
        for i in range(cfg.N_WORKERS):
            frame, reward, done = self.games[i].step(actions[i])
            if states is None:
                next_state = torch.cat([frame for i in range(cfg.AGENT_HISTORY_LENGTH)])
            else:
                next_state = torch.cat([states[i][1:], frame])
            next_state_list.append(next_state)
            reward_list.append(reward)
            done_list.append(done)

        return torch.stack(next_state_list), reward_list, done_list


    def train(self):
        """
        Main training loop.
        """
        # Episode lengths
        episode_length = []
        eplen = 0

        # Initialize the environment and state (do nothing)
        initial_actions = np.zeros(cfg.N_WORKERS)
        states, _, _ = self.env_step(None, initial_actions)

        # Start a training episode
        for i in range(1, cfg.TRAIN_ITERATIONS):

            # Forward pass through the net
            values, actions, action_log_probs = self.net.act(states)

            # Perform action in environment
            next_states, rewards, dones = self.env_step(states, actions)
            masks = torch.FloatTensor([[0.0] if done else [1.0] for done in dones])
            eplen += 1

            # Save experience to buffer
            self.memory.append(
                Experience(states.data, actions.data, action_log_probs.data, values.data, rewards, masks)
            )

            # Perform optimization
            if i % cfg.BUFFER_UPDATE_FREQ == 0 or done:
                loss = self.optimize_model(next_state, done)
                # Reset memory
                self.memory = []

            # Log episode length
            if done:
                self.writer.add_scalar('episode_length', eplen, i)
                print(i, eplen)
                eplen = 0

            # Save network
            if i % cfg.SAVE_NETWORK_FREQ == 0:
                if not os.path.exists(cfg.EXPERIMENT_NAME):
                    os.mkdir(cfg.EXPERIMENT_NAME)
                torch.save(self.net.state_dict(), f'{cfg.EXPERIMENT_NAME}/{str(i).zfill(7)}.pt')

            # Write results to log
            if i % cfg.LOG_FREQ == 0:
                self.writer.add_scalar('loss', loss, i)

            # Move on to next state
            states = next_states



    def play_game():
        return None







if __name__ == '__main__':
    x = Agent()
    x.train()
