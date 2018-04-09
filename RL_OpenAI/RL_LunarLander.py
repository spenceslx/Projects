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

#Lunar Lander State Vector:
#[pos_x, pos_y, vel_x, vel_y, angle, vel_angle, left_leg_contact, right_leg_contact]

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
        x = F.relu(self.l3(x))
        return x

def predict_action(var):
    """ Converts Net output Variable from a "continous onehot encoding" into
        an integer for action determination
        -- input:
            - var: The policy function output, which is a Net output,
                   a torch.autograd.Variable
        -- output, for LunarLander:
            - 0: Do nothing
            - 1: fire_left
            - 2: fire_right
            - 3: fire_main
    """
    _, indicies = var.data.max(0)
    #return a max action if there are more than one
    action = indicies[randint(0, len(indicies)-1)]
    return action


def train(policy, expected_reward):
    """ Trains a policy to take an action giving an environment observation,
        with the goal of maximizing the rewards from each trial
        -- input:
            - policy: A neural net of class Net
            - expected_reward: A neural net of class Net
        -- output:
            - policy: A trained neural net of class Net
            - expected_reward: A trained neural net of class Net
    """
    env = gym.make('LunarLander-v2')
    criterion = nn.MSELoss()
    for episode in range(0, num_episode):
        state = env.reset()
        frame, cum_reward, grad_policy, grad_reward = 0, 0, 0, 0
        isterminal = False
        state_store = [0 for x in range(0, max_frames)]
        action_store = [0 for x in range(0, max_frames)]
        reward_store = [0 for x in range(0, max_frames)]
        while (isterminal != True) or (frame < max_frames):
            action = policy(state)
            state, reward, isterminal, more_info = env.step(action)
            state_store[frame] = state
            action_store[frame] = action
            reward_store[frame] = reward
            frame += 1
        if isterminal == True:
            cum_reward = expected_reward[reward] #Bootstrap from last state
        #now accumulate gradients, rewards
        while frame > 0:
            cum_reward = reward_store[timestamp] + (discount_factor*cum_reward)
            grad_policy = grad_policy +
            frame -= 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=\
                        'Reinforcement Learning on LunarLander')
    parser.add_argument('mode', type=string, help='Train, test, or demo')
    parser.add_argument('--num_episodes', type=int, default=2000,\
                        help='Number of episodes (env.resets())to train on')
    parser.add_argument('--max_frames', type=int, default=1000,\
                        help='Maximum number of frames in the environment')
    parser.add_argument('--discount_factor', type=float32, default=0.99,\
                        help='Discount of previous rewards (gamma)')
    args = parser.parse_args()

    if mode == 'train':
        #environment observation is np.array length 8
        #environment step is np.array length 4
        policy = Net(n_feature=8, n_out=4, n_hidden=10)
        expected_reward = Net(n_feature=8, n_out=1, n_hidden=10)
        policy = train(policy, expected_reward)
