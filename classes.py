import torch.nn as nn
from collections import deque
import random
import torch
import numpy as np


class ShallowNet(nn.Module):
    """Fully connected neural net used to model the q-values"""
    def __init__(self, l1,l2,l3,l4):
        super().__init__()
        self.layers = nn.Sequential(
        nn.Linear(l1,l2),
        nn.ReLU(),
        nn.Linear(l2,l3),
        nn.ReLU(),
        nn.Linear(l3,l4))

    def forward(self, x):
        out = self.layers(x)
        return out


class ExperienceReplay:
    """Experience replay module to stabilize training"""
    def __init__(self, mem_size=1000, batch_size=200):
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.replay = deque(maxlen=self.mem_size)

    def add_to_memory(self, exp):
        self.replay.append(exp)

    def sample_batch(self):
        if len(self.replay) > self.batch_size:
            minibatch = random.sample(self.replay, self.batch_size)
            state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
            state2_batch = torch.cat([s2 for (s1, a, r, s2, d) in minibatch])
            action_batch = torch.Tensor([a for (s1, a, r, s2, d) in minibatch])
            reward_batch = torch.Tensor([r for (s1, a, r, s2, d) in minibatch])
            done_batch = torch.Tensor([d for (s1, a, r, s2, d) in minibatch])
            return state1_batch, action_batch, reward_batch, state2_batch, done_batch
        else:
            raise ValueError("Replay size must be greater than Batch size")

    def __len__(self):
        return len(self.replay)


class Logger:
    """Logs different kinds of data during training (e.g. losses)"""
    def __init__(self):
        self.losses = []
        self.tot_rewards = []
        self.game_lengths = []
        self.game_outcomes = []
        self.m_outcomes = []
        self.lost_games = []
        self.m_lost_games = []

    def log(self, loss, tot_reward, game_length, reward):
        self.losses.append(loss)
        self.tot_rewards.append(tot_reward)
        self.game_lengths.append(game_length)
        if reward == 6:
            self.game_outcomes.append(1)
        else:
            self.game_outcomes.append(0)
        if len(self.game_outcomes)>100:
            self.m_outcomes.append(np.mean(self.game_outcomes[-100:-1]))
        if reward == -5:
            self.lost_games.append(1)
        else:
            self.lost_games.append(0)
        if len(self.lost_games)>100:
            self.m_lost_games.append(np.mean(self.lost_games[-100:-1]))
