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
        self.softmax1 = torch.nn.Softmax()
        self.logsoftmax1 = torch.nn.LogSoftmax()


    def init_weights(self, m):
        """
        Initialize the weights of the network.

        Arguments:
            m (tensor): layer instance 
        """
        if type(m) == torch.nn.Conv2d or type(m) == torch.nn.Linear:
            torch.nn.init.uniform(m.weight, -0.01, 0.01)
            m.bias.data.fill_(0.01)


    def forward(self, x, eps=-1):
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
        b = x.shape[0]
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = x.view(x.size()[0], -1)
        x = self.relu3(self.fc3(x))
        y = self.actor(x)
        action_probs = torch.squeeze(self.softmax1(y))
        log_action_probs = torch.squeeze(self.logsoftmax1(y))
        value = torch.squeeze(self.critic(x))

        # Choose action
        action = Categorical(action_probs).sample(b).detach()
        if random.random() <= eps:
            action = np.random.choice(cfg.N_ACTIONS, p=[0.95, 0.05], size=b)
        action = torch.tensor(action, dtype=torch.long)
        if b > 1:
            action = action.unsqueeze(-1)

        # Calculate auxiliary values
        entropy = -(action_probs*log_action_probs).sum(axis=-1) 
        log_prob = log_action_probs.gather(-1, action)
        return action.squeeze(), entropy, log_prob.squeeze(), value



class Agent():

    def __init__(self):
        """
        Initialize an A2C Instance. 
        """
        # Create ACNetwork
        self.net = ActorCriticNetwork()
        self.net.apply(self.net.init_weights)

        # Optimizer
        self.opt = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

        # The flappy bird game instance
        self.game = Game(cfg.FRAME_SIZE)

        # Log to tensorBoard
        self.writer = SummaryWriter(cfg.EXPERIMENT_NAME)

        # Loss
        self.loss = torch.nn.MSELoss()

        # Buffer
        self.buffer = {k: [] for k in ['value', 'action_prob', 'reward', 'entropy']}
        self.buffer_length = 0

        # Epsilon used for selecting actions
        self.epsilon = np.linspace(
            cfg.INITIAL_EXPLORATION, 
            cfg.FINAL_EXPLORATION, 
            cfg.FINAL_EXPLORATION_FRAME
        )


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
        if done:
            value_1 = 0
        else:
            batch_next_state = next_state.unsqueeze(0)
            _, _, _, value_1 = self.net(batch_next_state)
        self.buffer['value'].append(value_1)

        # Calculate the losses
        loss_value, loss_policy, gae = 0, 0, 0
        for i in reversed(range(self.buffer_length)):

            value_target = cfg.DISCOUNT * value_1 + self.buffer['reward'][i]
            advantage = value_1 - self.buffer['value'][i]
            loss_value += advantage ** 2 

            delta_t = self.buffer['reward'][i] + cfg.DISCOUNT*self.buffer['value'][i+1] - self.buffer['value'][i]
            gae = gae * cfg.DISCOUNT + delta_t
            loss_policy -= self.buffer['action_prob'][i] * gae.detach() - cfg.ENTROPY_COEFF * self.buffer['entropy'][i]

        # Total loss
        loss = loss_policy + cfg.VALUE_LOSS_COEFF * loss_value

        # Push local gradients to global network
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.MAX_GRAD_NORM)
        for lp, gp in zip(self.net.parameters(), self.global_net.parameters()):
            gp._grad = lp.grad
        self.opt.step()

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
            batch_state = state.unsqueeze(0)
            eps = self.epsilon[min(i, cfg.FINAL_EXPLORATION_FRAME - 1)]
            action, entropy, log_prob, value = self.net(batch_state, eps=eps)

            # Perform action in environment
            frame, reward, done = self.game.step(action)

            # Save experience to buffer
            self.buffer['value'].append(value)
            self.buffer['action_prob'].append(log_prob)
            self.buffer['reward'].append(reward)
            self.buffer['entropy'].append(entropy)
            self.buffer_length += 1

            # Update next state
            next_state = torch.cat([state[1:], frame])
            eplen += 1

            # Perform optimization
            # Update global and local networks
            if self.buffer_length % cfg.BUFFER_UPDATE_FREQ == 0 or done:
                loss = self.optimize_model(next_state, done)

                # Clear buffers
                self.buffer = dict.fromkeys(self.buffer, [])
                self.buffer_length = 0

                if done:
                    self.writer.add_scalar('episode_length', eplen, i)
                    print(i, eplen)
                    eplen = 0

            # Save network
            if i % cfg.SAVE_NETWORK_FREQ == 0:
                if not os.path.exists(cfg.EXPERIMENT_NAME):
                    os.mkdir(cfg.EXPERIMENT_NAME)
                torch.save(self.global_net.state_dict(), f'{cfg.EXPERIMENT_NAME}/{str(i).zfill(7)}.pt')

            # Write results to log
            if i % 100 == 0:
                self.writer.add_scalar('loss', loss, i)

            # Move on to next state
            state = next_state


    def play_game():
        return None



class Trainer():
    def __init__(self):
        """
        Create the training instance which will coordinate all the individual
        ActorCriticWorker subprocesses.
        """

        # Global shared network
        self.net = ActorCriticNetwork()
        self.net.share_memory()

        # Optimizer
        self.opt = SharedAdam(self.net.parameters())
        self.opt.share_memory()

        # Create the workers
        torch.multiprocessing.set_start_method('spawn')
        self.workers = [
            torch.multiprocessing.Process(
                target=actor_critic_worker_entrypoint, 
                args=(i, self.net, self.opt)
            )  
            for i in range(cfg.N_WORKERS)
        ] 

    def train(self):
        """
        Start the global training loop
        """
        # Start the workers
        for worker in self.workers:
            worker.start()
        
        # Finish jobs
        for worker in self.workers:
            worker.join()



if __name__ == '__main__':
    x = Trainer()
    x.train()
