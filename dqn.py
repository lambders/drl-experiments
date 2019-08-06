"""
Implementation of Deep Q-Networks by the Google Brain team.
Reference:
    "Human-Level Control Through Deep Reinforcement Learning" by Mnih et al. 
"""

import os
import math
import random
import torch
import numpy as np 
from tensorboardX import SummaryWriter
from collections import namedtuple

import config as cfg
from game.wrapper import Game 

# TODO: Play loop
# TODO: Plot training episode times/cumulative reward

class DQN(torch.nn.Module):

    def __init__(self):
        """
        Initialize a Deep Q-Network instance.
        Uses the same parameters as specified in the paper.
        """
        super(DQN, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(cfg.AGENT_HISTORY_LENGTH, 32, 8, 4)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(32, 64, 4, 2)
        self.relu2 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(64, 64, 3, 1)
        self.relu3 = torch.nn.ReLU()
        self.fc4 = torch.nn.Linear(3136, 512) # TODO: Don't hard code
        self.relu4 = torch.nn.ReLU()
        self.fc5 = torch.nn.Linear(512, cfg.N_ACTIONS)


    def forward(self, x):
        """
        Forward pass to compute Q-values for given input states.

        Arguments:
            x (tensor): minibatch of input states

        Returns:
            tensor: state-action values of size (batch_size, n_actions)
        """
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = x.view(x.size()[0], -1)
        x = self.relu4(self.fc4(x))
        x = self.fc5(x)
        return x



Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory():

    def __init__(self):
        """
        Initialize a replay memory instance.
        Used by agent to create minibatches of experiences. Resuts in greater 
        data efficiency, reduced update variance, and smoother learning.
        """
        self.memory = []
        self.capacity = cfg.REPLAY_MEMORY_SIZE


    def add(self, experience):
        """
        Add an experience to replay memory.

        Arguments:
            experience (Experience): add experience to replay memory 
        """
        self.memory.append(experience)

        # Remove oldest experience if replay memory full
        if len(self.memory) > self.capacity:
            replay_memory.pop(0)


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

        return {
            'state': torch.stack(sample.state).to(cfg.DEVICE),
            'action': torch.tensor(sample.action).unsqueeze(1).to(cfg.DEVICE),
            'reward': torch.tensor(sample.reward).to(cfg.DEVICE),
            'next_state': torch.stack(sample.next_state).to(cfg.DEVICE),
            'done': torch.tensor(sample.done).to(cfg.DEVICE),
        }


class Agent:

    def __init__(self):
        """
        Initialize an agent instance.
        """

        # Replay memory buffer
        self.replay_memory = ReplayMemory()

        # Epsilon used for selecting actions
        self.epsilon = np.logspace(
            math.log(cfg.INITIAL_EXPLORATION), 
            math.log(cfg.FINAL_EXPLORATION), 
            num=cfg.FINAL_EXPLORATION_FRAME, 
            base=math.e
        )

        # Create policy and target DQNs
        self.target_net = DQN().to(cfg.DEVICE)
        self.policy_net = DQN().to(cfg.DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # The optimizer
        # self.optimizer = torch.optim.RMSprop(
        #     params = self.policy_net.parameters(),
        #     lr = cfg.LEARNING_RATE,
        #     momentum = cfg.GRADIENT_MOMENTUM,
        #     alpha = cfg.SQUARED_GRADIENT_MOMENTUM,
        #     eps = cfg.MIN_SQUARED_GRADIENT
        # )
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
                return torch.argmax(self.policy_net(state), dim=1)[0]


    def optimize_model(self):
        """
        Performs a single step of optimization.
        Samples a minibatch from replay memory and uses that to update the policy_net.


        Returns:
            loss (float)
        """
        # Sample a batch [state, action, reward, next_state]
        batch = self.replay_memory.sample(cfg.MINIBATCH_SIZE)
        if batch is None:
            return

        # Compute Q(s_t, a) using the policy_net. 
        q_batch = torch.gather(self.policy_net(batch['state']), 1, batch['action'])
        q_batch = q_batch.squeeze()

        # Compute V(s_{t+1}) for all next states using the target_net. 
        # Using a network with an older set of parameters adds a delay between 
        # the time an update to the network is made and the time the update 
        # affects targets y, stabilizing training
        q_batch_1, _ = torch.max(self.target_net(batch['next_state']), dim=1)
        y_batch = batch['reward'] + cfg.DISCOUNT_FACTOR * q_batch_1 
        y_batch = torch.tensor(
            [batch['reward'][i] if batch['done'][i] else y_batch[i] for i in range(cfg.MINIBATCH_SIZE)]
            )

        # Compute loss
        loss = self.loss(q_batch, y_batch.to(cfg.DEVICE))

        # Optimize model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss


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

            # Save experience to replay memory
            self.replay_memory.add(
                Experience(state, action, reward, next_state, done)
            )

            # Perform optimization
            # Sample random minibatch and update policy network
            loss = self.optimize_model()

            # Move on to the next state
            state = next_state

            # Update the target network
            if i % cfg.TARGET_NETWORK_UPDATE_FREQ == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # Save network
            if i % cfg.SAVE_NETWORK_FREQ == 0:
                if not os.path.exists('results'):
                    os.mkdir('results')
                torch.save(self.target_net.state_dict(), f'results/{str(i).zfill(7)}.pt')

                np.save('eplen.npy', episode_length)

            # Write results to log
            if i % 100 == 0:
                self.writer.add_scalar('loss', loss, i)

            eplen += 1
            if done:
                print(i, eplen)
                self.writer.add_scalar('episode_length', eplen, i)
                eplen = 0


    def play_game():
        return None


if __name__ == '__main__':
    x = Agent()
    x.train()

