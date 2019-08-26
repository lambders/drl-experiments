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

import config.a3c as cfg
from game.wrapper import Game 

# Global parameter which tells us if we have detected a CUDA capable device
CUDA_DEVICE = torch.cuda.is_available()



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
        self.fc3 = torch.nn.Linear(2592, 256) # TODO: Don't hard code
        self.relu3 = torch.nn.ReLU()
        self.actor = torch.nn.Linear(256, cfg.N_ACTIONS)
        self.critic = torch.nn.Linear(256, 1)
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
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = x.view(x.size()[0], -1)
        x = self.relu3(self.fc3(x))

        y = self.actor(x)
        action_probs = self.softmax1(y)
        log_action_probs = self.logsoftmax1(y)
        values = self.critic(x)
        # torch.Size([1, 2]) torch.Size([1, 1])
        return action_probs[0], log_action_probs[0], values[0][0]



class SharedAdam(torch.optim.Adam):

    def __init__(self, params, lr=cfg.LEARNING_RATE):
        """
        Initialize a shared Adam optimizer.
        Taken from ikostrikov's repo:
        https://github.com/ikostrikov/pytorch-a3c/blob/master/my_optim.py

        Arguments:
            params: network parameters to optimize
            lr (float): learning rate
        """
        super(SharedAdam, self).__init__(params, lr)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()

    def share_memory(self):
        """
        Share memory globally.
        """
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step'].item()
                bias_correction2 = 1 - beta2 ** state['step'].item()
                step_size = group['lr'] * math.sqrt(
                    bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss



class ActorCriticWorker(torch.multiprocessing.Process):

    def __init__(self, global_net, shared_opt):
        """
        Initialize an actor-critic worker. 

        Arguments:
            global_net (torch.nn.Module): global network this instance will 
                update periodically
            shared_optim (SharedAdam): global optimizer shared by all 
                of the workers
        """
        # Global network
        self.global_net = global_net

        # Shared optimizer
        self.shared_opt = shared_opt

        # Create local ACNetwork
        self.net = ActorCriticNetwork()
        if CUDA_DEVICE:
            self.net = self.net.cuda()

        # The flappy bird game instance
        self.game = Game(cfg.FRAME_SIZE)

        # Log to tensorBoard
        self.writer = SummaryWriter(cfg.EXPERIMENT_NAME)

        # Loss
        self.loss = torch.nn.MSELoss()

        # Buffer
        self.buffer = {k: [] for k in ['state', 'action', 'reward']}


    def select_action(self, state):
        """
        Select the next action. 

        Arguments:
            state (tensor): stack of four frames

        Returns:
            int: 0 if no flap, 1 if flap
        """
        # Set to evaluation mode
        self.net.eval()

        # Make state have a batch size of 1
        state = state.unsqueeze(0)
        if CUDA_DEVICE:
            state = state.cuda()

        # Select action according to probability
        action_probs = self.net(state)[0].detach().numpy()
        selected_action = np.random.choice(
            np.arange(cfg.N_ACTIONS), 1, p=action_probs)

        # Set back to train mode
        self.net.train()

        return selected_action


    def optimize_model(self, next_state, done):
        """
        Performs a single step of optimization.
        Samples a minibatch from replay memory and uses that to update the net.

        Arguments:
            next_state (tensor): next frame of the game
            done (bool): True if next_state is a terminal state, else False

        Returns:
            loss (float)
        """
        # Forward pass through the net
        action_probs, log_action_probs, values = self.net(next_state)

        # Calculate the value of the next state
        if done:
            value_1 = 0
        else:
            value_1 = values[-1] 

        # Calculate discounted reward
        reward = self.buffer['reward']
        discounts = [self.GAMMA**i for i in range(len(reward))][::-1]
        value_target = [r + gamma*value_1 for (r, gamma) in zip(reward, discounts)]

        # Calculate value loss
        advantage = value_target - value_1
        loss_value = advantage ** 2

        # Calculate policy loss
        entropy = action_probs * log_action_probs
        loss_policy = -log_action_probs * advantage - cfg.ENTROPY_REG * entropy

        # Total loss
        loss = loss_policy + cfg.VALUE_LOSS_WEIGHT * loss_value

        # Calculate local gradients and push local parameters to global
        self.optimizer.zero_grad()
        loss.backward()
        for lp, gp in zip(self.net.parameters(), gnet.parameters()):
            gp._grad = lp.grad
        self.optimizer.step()

        # pull global parameters
        self.net.load_state_dict(gnet.state_dict())

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
        state = torch.cat([frame for i in range(cfg.AGENT_HISTORY_LENGTH)])

        # Start a training episode
        for i in range(1, cfg.TRAIN_ITERATIONS):

            # Perform an action
            action = self.select_action(state)
            frame, reward, done = self.game.step(action)
            next_state = torch.cat([state[1:], frame])

            # Save experience to buffer
            self.buffer['state'].append(state)
            self.buffer['action'].append(action)
            self.buffer['reward'].append(reward)

            # Perform optimization
            # Sample random minibatch and update policy network
            loss = self.optimize_model(next_state, done)

            # Update global and local networks
            if i % cfg.GLOBAL_NET_UPDATE_FREQ == 0 or done: # TODO:  or done?
                self.synchronize_nets(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)

            # Write results to log
            if i % 100 == 0:
                self.writer.add_scalar('loss', loss, i)

            eplen += 1
            if done:
                print(i, eplen, reward, done)
                self.writer.add_scalar('episode_length', eplen, i)
                eplen = 0

            # Move on to next state
            state = next_state


    def play_game():
        return None



def Trainer():
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
        self.opt.shared_memory()

        # Start the workers
        self.workers = [ActorCriticWorker(target=ActorCriticWorker.train, 
            args=(self.net, self.opt)) for i in range(cfg.N_WORKERS)] 


    def train(self):
        """
        Start the global training loop
        """
        # Start the workers
        for worker in self.workers:
            worker.start()

        # Parallel training
        res = []
        while True:
            r = results_queue.get()
            if r is not None:
                res.append(r)
            else:
                break
        [w.join() for w in workers]



if __name__ == '__main__':
    # x = ActorCriticWorker(None, None)
    # x.train()
