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
        self.fc3 = torch.nn.Linear(3136, 256) # TODO: Don't hard code
        self.relu3 = torch.nn.ReLU()
        self.actor = torch.nn.Linear(256, cfg.N_ACTIONS)
        self.crtic = torch.nn.Linear(256, 1)
        self.softmax1 = torch.nn.SoftMax()


    def forward(self, x):
        """
        Forward pass to compute Q-values for given input states.

        Arguments:
            x (tensor): minibatch of input states

        Returns:
            tensor: prob of selecting an action, of size (batch_size, n_actions)
            tensor: single linear output representing the value fn (batch_size, 1)
        """
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = x.view(x.size()[0], -1)
        x = self.relu3(self.fc3(x))
        probability = self.softmax1(self.actor(x))
        value = self.critic(x)
        return probability, value


class SharedAdam(torch.optim.Adam):
    def __init__(self, params):
        """
        Initialized a shared Adam optimizer instance.
        All the ActorCritic threads will subscribe to this optimizer and update
        in an asynchronous manner.

        Arguments:
            params (): network parameters to optimize
        """
        super(SharedAdam, self).__init__(params, lr=cfg.LEARNING_RATE)

        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()



class ActorCriticWorker(torch.multiprocessing.Process):

    def __init__(self):
        """
        Initialize an actor-critic worker. 
        """

        # Epsilon used for selecting actions
        self.epsilon = np.linspace(
            cfg.INITIAL_EXPLORATION, 
            cfg.FINAL_EXPLORATION, 
            cfg.FINAL_EXPLORATION_FRAME
        )

        # Create local ACNetwork
        self.net = ActorCriticNetwork()
        if CUDA_DEVICE:
            self.net = self.net.cuda()

        # The optimizer
        # self.optimizer = torch.optim.Adam(
        #     self.policy_net.parameters(),
        #     lr = cfg.LEARNING_RATE
        # )

        # The flappy bird game instance
        self.game = Game(cfg.FRAME_SIZE)

        # Log to tensorBoard
        self.writer = SummaryWriter(cfg.EXPERIMENT_NAME)

        # Loss
        self.loss = torch.nn.MSELoss()

        # Buffer
        self.buffer = {'state': [], 'action': [], 'reward': [], 'value': []}


    def select_action(self, state):
        """
        Select the next action. 

        Arguments:
            state (tensor): stack of four frames

        Returns:
            int: 0 if no flap, 1 if flap
        """
        # Make state have a batch size of 1
        state = state.unsqueeze(0)
        if CUDA_DEVICE:
            state = state.cuda()

        # Select action according to probability
        prob, _ = self.net(state)[0]
        return np.random.choice(np.arange(cfg.N_ACTIONS, 1, p=prob))


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

            # Save network
            # if i % cfg.SAVE_NET_FREQ == 0:
            #     if not os.path.exists('results'):
            #         os.mkdir('results')
            #     torch.save(self.target_net.state_dict(), f'results/{str(i).zfill(7)}.pt')
            # Update global network
            if i % cfg.GLOBAL_HET_UPDATE_FREQ == 0:
                push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)

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
        self.optimizer = SharedAdam(self.net.parameters())

        # Start the workers
        self.workers = [ActorCriticWorker() for i in range(cfg.N_WORKERS)] 


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
    x = Trainer()
    x.train()
