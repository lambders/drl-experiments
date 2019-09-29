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

# TODO: Log

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



class ActorCriticWorker():

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
        # Worker id
        self.id = id

        # Global network
        self.global_net = global_net

        # Shared optimizer
        self.shared_opt = shared_opt

        # Create local ACNetwork
        self.net = ActorCriticNetwork()

        # Synchronize with shared model
        self.net.load_state_dict(self.global_net.state_dict())

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

            # Calculate entropy
            entropy = -torch.sum(action_probs * log_action_probs)

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
            if self.buffer_length % cfg.BUFFER_UPDATE_FREQ == 0 or done:
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
            if done and id == 0:
                print(self.id, i, eplen)
                self.writer.add_scalar('episode_length/' + str(self.id), eplen, i)
                eplen = 0

            # Move on to next state
            state = next_state


    def play_game():
        return None



def actor_critic_worker_entrypoint(id, net, opt):
    """
    The entrypoint for a torch multiprocess. 
    Must be at top level.

    Args:
        id (int): worker id
        net (torch.nn.Module): shared network isntance
        opt (torch.optim): shared optimizer
    """
    worker = ActorCriticWorker(id, net, opt)
    worker.train()



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
