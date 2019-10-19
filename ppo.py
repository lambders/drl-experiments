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

import config_ppo as cfg
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
            torch.nn.init.uniform(m.weight, -0.5, 0.5)
            m.bias.data.fill_(0.01)


    def forward(self, x, action=None, eps=-1):
        """
        Forward pass to compute Q-values for given input states.

        Arguments:
            x (tensor): minibatch of input states
            action (int, optional): 0 to not flap, 1 to flap
            eps (float, optional): choose random action with eps probability

        Returns:
            int: selected action, 0 to do nothing and 1 to flap
            float: entropy of action space
            float: log probability of selecting the action 
            float: value of the particular state
        """
        try:
            # Forward pass
            b = x.shape[0]
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.relu2(self.conv2(x))
            x = x.view(x.size()[0], -1)
            x = self.relu3(self.fc3(x))
            y = self.actor(x)
            action_probs = torch.squeeze(self.softmax1(y))
            log_action_probs = torch.squeeze(self.logsoftmax1(y))
            value = torch.squeeze(self.critic(x))

            # Choose action
            if action is None:
                action = Categorical(action_probs).sample_n(b).detach()
                if random.random() <= eps:
                    action = np.random.choice(cfg.N_ACTIONS, p=[0.95, 0.05], size=b)
            action = torch.tensor(action, dtype=torch.long)
            if b > 1:
                action = action.unsqueeze(-1)

            # Calculate auxiliary values
            entropy = -(action_probs*log_action_probs).sum(axis=-1) 
            log_prob = log_action_probs.gather(-1, action)
            return action.squeeze(), entropy, log_prob.squeeze(), value
        except RuntimeError as e:
            print(e)
            return None, None, None, None



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
        self.global_net.apply(self.global_net.init_weights)

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

        # Buffer
        self.buffer = {k: [] for k in ['value', 'action_prob', 'reward', 'state' ,'action']}
        self.buffer_length = 0

        # Epsilon used for selecting actions
        self.epsilon = np.linspace(
            cfg.INITIAL_EXPLORATION, 
            cfg.FINAL_EXPLORATION, 
            cfg.FINAL_EXPLORATION_FRAME
        )


    def optimize_model(self):
        """
        Performs a single step of optimization.

        Arguments:

        Returns:
            loss (float)
        """
        # Converting the buffers to tensors
        buffer_action_prob = torch.Tensor(self.buffer['action_prob']).detach()
        buffer_reward = torch.Tensor(self.buffer['reward']).detach()
        # buffer_reward = (buffer_reward - buffer_reward.mean()) / (buffer_reward.std() + 1e-5)
        buffer_value = torch.Tensor(self.buffer['value']).detach()
        buffer_action = torch.Tensor(self.buffer['action']).detach()
        buffer_state = torch.stack(self.buffer['state']).detach()

        for i in range(cfg.UPDATE_EPOCHS):

            # Evaluating old actions and values:
            _, entropy, log_prob, value = self.net(buffer_state, buffer_action)
            
            # Action loss
            ratio = torch.exp(log_prob - buffer_action_prob)
            adv = buffer_reward - value
            adv = (adv - adv.mean()) / (adv.std() + 1e-5)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1-cfg.CLIP, 1+cfg.CLIP) * adv
            action_loss = -torch.min(surr1, surr2).mean()

            # Value Loss
            value_clipped = buffer_value + (value - buffer_value).clamp(-cfg.CLIP, cfg.CLIP)
            value_losses = (value - buffer_reward).pow(2)
            value_losses_clipped = (value_clipped - buffer_reward).pow(2)
            value_loss = cfg.VALUE_LOSS_COEFF * torch.max(value_losses, value_losses_clipped).mean()
            # print(value.shape, buffer_reward.shape)
            # print(torch.nn.MSELoss(value, buffer_reward))
            # value_loss = cfg.VALUE_LOSS_COEFF * torch.nn.MSELoss(value, buffer_reward)
            
            # Entropy loss
            entropy_loss = cfg.ENTROPY_COEFF * torch.mean(entropy)
            
            # Push local gradients to global network
            loss = action_loss + value_loss - entropy_loss
            self.shared_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), cfg.MAX_GRAD_NORM)
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
            batch_state = state.unsqueeze(0)
            eps_step = min(i, cfg.FINAL_EXPLORATION_FRAME - 1)
            action, entropy, log_prob, value = self.net(batch_state, eps=self.epsilon[eps_step])
            if action is None:
                self.game.restart()
                eplen = 0
                print("RESTART@")
                continue
            
            # Perform action in environment
            frame, reward, done = self.game.step(action)

            # Save experience to buffer
            self.buffer['state'].append(state)
            self.buffer['action'].append(action)
            self.buffer['value'].append(value)
            self.buffer['action_prob'].append(log_prob)
            self.buffer['reward'].append(reward)
            self.buffer_length += 1

            # Update next state
            next_state = torch.cat([state[1:], frame])
            eplen += 1

            # Perform optimization
            # Update global and local networks
            if self.buffer_length % cfg.BUFFER_UPDATE_FREQ == 0 or done:
                loss = self.optimize_model()

                # Clear buffers
                self.buffer = {k: [] for k in ['value', 'action_prob', 'reward', 'state' ,'action']}
                self.buffer_length = 0

                if done:
                    if self.id == 2:
                        self.writer.add_scalar('episode_length/' + str(self.id), eplen, i)
                    print(self.id, i, eplen)
                    eplen = 0

                # # Initialize the environment and state (do nothing)
                # frame, reward, done = self.game.step(0)
                # state = torch.cat([frame for i in range(cfg.AGENT_HISTORY_LENGTH)])


            # Save network
            if i % cfg.SAVE_NETWORK_FREQ == 0:
                if not os.path.exists(cfg.EXPERIMENT_NAME):
                    os.mkdir(cfg.EXPERIMENT_NAME)
                torch.save(self.global_net.state_dict(), f'{cfg.EXPERIMENT_NAME}/{str(i).zfill(7)}.pt')

            # Write results to log
            if i % 100 == 0:
                if self.id == 2:
                    self.writer.add_scalar('loss/'+ str(self.id), loss, i)

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
