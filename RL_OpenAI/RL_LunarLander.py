#Python3
import gym
import Box2D
import torch
import torchvision
import argparse
from time import time
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """ Building 3 linear layer net to give an action given an environment
        -- input:
            - n_feature: number of input features (int)
            - n_out: number of output features (int)
            - n_hidden (optional): number nodes in hidden layer (int)
        -- output: a nn.autograd.Variable of size n_out
    """
    def __init__(self, n_feature, n_out, n_hidden=10):
        super(Net, self).__init__()
        self.l1 = nn.Linear(n_feature, n_hidden)
        self.l2 = nn.Linear(n_hidden, n_hidden)
        self.l3 = nn.Linear(n_hidden, n_out)
        self.n_feature = n_feature
        self.n_hidden = n_hidden
        self.n_out = n_out

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.sigmoid(self.l3(x)) #[0,1] activation

def predict_action(var):
    """ Converts Net output Variable from a "continous onehot encoding" into
        an integer for action determination
        -- input:
            - var: The policy function output, which is a Net output,
                   a torch.autograd.Variable
        -- output:
            - 0: Do nothing
            - 1: fire_left
            - 2: fire_right
            - 3: fire_main
    """
    _, indicies = var.data.max(0)
    #return a max action if there are more than one
    action = indicies[randint(0, len(indicies)-1)]
    return action


def train(policy, num_episodes=num_episodes, max_frames=max_frames):
    """ Trains a policy to take an action giving an environment observation,
        with the goal of maximizing the rewards from each trial
        -- input:
            - policy: A neural net of class Net
            - num_episodes: Number of episodes to train for (env.reset()'s)
            - max_frames: Maximum number of frames for an episode to run for
        -- output:
            - policy: A trained neural net of class Net
    """
    env = gym.make('LunarLander-v2')
    criterion = nn.MSELoss()
    for episode in range(0, num_episode):
        observation = env.reset()
        for frame (0, max_frames):



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=\
                        'Reinforcement Learning on LunarLander')
    parser.add_argument('mode', type=string, help='train, test, or demo')
    parser.add_argument('--num_episodes', type=int, default=2000,\
                        help='Number of episodes (env.resets())to train on')
    parser.add_argument('--max_frames', type=int, default=1000,\
                        help='Maximum number of frames in the environment')
    args = parser.parse_args()

    if mode == 'train':
        #environment observation is np.array length 8
        #environment step is np.array length 4
        n_feature = 8
        n_out = 4
        n_hidden = 10
        policy = Net(n_feature, n_out, n_hidden)
        policy = train(Net)
