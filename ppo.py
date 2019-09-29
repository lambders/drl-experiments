"""
Implementation of PPO by the OpenAI team.
Reference:
    "Proximal Policy Optimization Algorithms" by Klimov et al. 
"""

import os
import math
import random
import torch
import numpy as np 
from tensorboardX import SummaryWriter
from collections import namedtuple

import config.ppo as cfg
from game.wrapper import Game 

# TODO: Log

class ActorCriticNetwork(torch.nn.Module):

    def __init__(self):
        """
        Initialize an ActorCriticNetwork instance.
        Uses the same parameters as specified in the paper.
        """
        super(ActorCriticNetwork, self).__init__()
        
        self.conv1 = torch.nn.Linear(cfg.AGENT_HISTORY_LENGTH, 32, 8, 4)
        self.tanh1 = torch.nn.Tanh()
        self.conv2 = torch.nn.Conv2d(32, 64, 4, 2)
        self.tanh2 = torch.nn.Tanh()
        self.conv3 = torch.nn.Conv2d(64, 64, 3, 1)
        self.tanh3 = torch.nn.Tanh()
        self.fc4 = torch.nn.Linear(3136, 512) 
        self.tanh4 = torch.nn.Tanh()
        self.actor = torch.nn.Linear(512, cfg.N_ACTIONS)
        self.critic = torch.nn.Linear(512, 1)
        self.softmax1 = torch.nn.Softmax()
        self.logsoftmax1 = torch.nn.LogSoftmax()


    def forward(self, x):
        """
        Forward pass to compute Q-values for given input states.

        Arguments:
            x (tensor): minibatch of input states

        Returns:
            tensor: prob of selecting an action, of size (n_actions)
            tensor: single linear output representing the value fn (1)
        """
        x = self.tanh1(self.conv1(x))
        x = self.tanh2(self.conv2(x))
        x = self.tanh3(self.conv3(x))
        x = x.view(x.size()[0], -1)
        x = self.tanh4(self.fc4(x))

        y = self.actor(x)
        action_probs = self.softmax1(y)
        log_action_probs = self.logsoftmax1(y)
        values = self.critic(x)
        # torch.Size([1, 2]) torch.Size([1, 1])
        return action_probs[0], log_action_probs[0], values[0][0]



class Agent():

    def __init__(self, id, global_net, shared_opt):
        """
        Initialize an actor-critic worker. 

        Arguments:
            id (int): worker id
            global_net (torch.nn.Module): global network this instance will 
                update periodically
            shared_optim (SharedAdam): global optimizer shared by all 
                of the workers
        """
        # Create train ACNetwork
        self.net = ActorCriticNetwork()

        # Create stable network
        self.net_stable = ActorCriticNetwork()
        self.net_stable.load_state_dict(self.net.state_dict())

        # The flappy bird game instance
        self.game = Game(cfg.FRAME_SIZE)

        # Log to tensorBoard
        self.writer = SummaryWriter(cfg.EXPERIMENT_NAME)

        # Loss
        self.loss = torch.nn.MSELoss()

        # Buffer
        self.buffer = {k: [] for k in ['value', 'action_prob', 'reward', 'entropy']}
        self.buffer_length = 0


    def optimize_model(self, next_state, done):
        """
        Performs a single step of optimization.

        Arguments:
            next_state (tensor): next frame of the game
            done (bool): True if next_state is a terminal state, else False

        Returns:
            loss (float)
        """

        # Forward pass through the net
        batch_next_state = next_state.unsqueeze(0)
        _, _, value_1 = self.net(batch_next_state)

        # Calculate the value of the next state
        if done:
            value_1 = 0

        # Calculate the losses
        loss_value, loss_policy = 0, 0
        value_target = value_1
        for i in reversed(range(self.buffer_length)):
            value_target = cfg.DISCOUNT * value_target + self.buffer['reward'][i]
            advantage = value_target - self.buffer['value'][i]

            loss_value += advantage ** 2 
            loss_policy += -self.buffer['action_prob'][i] * advantage - cfg.ENTROPY_COEFF * self.buffer['entropy'][i]

        # Total loss
        loss = loss_policy + cfg.VALUE_LOSS_COEFF * loss_value

        # Push local gradients to global network
        self.shared_opt.zero_grad()
        loss.backward()
        for lp, gp in zip(self.net.parameters(), self.global_net.parameters()):
            gp._grad = lp.grad
        self.shared_opt.step()

        # Pull global parameters
        self.net.load_state_dict(self.global_net.state_dict())

        return loss


    def train(self):
        """
        Main training loop.
        """
        # Episode lengths
        episode_length = []
        eplen = 0

        # Initialize the environment and state (do nothing)
        frame, reward, done = self.game.step(0)
        state = torch.cat([frame for i in range(cfg.AGENT_HISTORY_LENGTH)])

        # Start a training episode
        for i in range(1, cfg.TRAIN_ITERATIONS):

            # Forward pass through the net
            self.net.eval()
            batch_state = state.unsqueeze(0)
            action_probs, log_action_probs, value = self.net(batch_state)
            self.net.train()

            # Perform action according to action_probs
            selected_action = np.random.choice(
                np.arange(cfg.N_ACTIONS), 1, p=action_probs.detach().numpy())[0]
            selected_action_prob = log_action_probs[selected_action]
            frame, reward, done = self.game.step(selected_action)

            # Finding the policy ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
            # Update next state
            next_state = torch.cat([state[1:], frame])

            # Save experience to buffer
            self.buffer['value'].append(value)
            self.buffer['action_prob'].append(selected_action_prob)
            self.buffer['reward'].append(reward)
            self.buffer['entropy'].append(entropy)
            self.buffer_length += 1

            # Perform optimization
            # Update global and local networks
            if self.buffer_length % cfg.BUFFER_UPDATE_FREQ == 0:
                loss = self.optimize_model(next_state, done)

                # Clear buffers
                self.buffer = dict.fromkeys(self.buffer, [])
                self.buffer_length = 0

            # Save network
            if i % cfg.SAVE_NETWORK_FREQ == 0:
                if not os.path.exists(cfg.EXPERIMENT_NAME):
                    os.mkdir(cfg.EXPERIMENT_NAME)
                torch.save(self.global_net.state_dict(), f'{cfg.EXPERIMENT_NAME}/{str(i).zfill(7)}.pt')

            # Write results to log
            if i % 100 == 0:
                self.writer.add_scalar('loss/'+ str(self.id), loss, i)

            eplen += 1
            if done:
                print(self.id, i, eplen)
                self.writer.add_scalar('episode_length/' + str(self.id), eplen, i)
                eplen = 0

            # Move on to next state
            state = next_state


    def play_game():
        return None


if __name__ == '__main__':
    x = Trainer()
    x.train()
