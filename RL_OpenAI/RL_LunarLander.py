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
import torch.optim as optim

#Lunar Lander State Vector:
#[pos_x, pos_y, vel_x, vel_y, angle, vel_angle, left_leg_contact, right_leg_contact]

class ActorCritic(nn.Module):
    """ Creates a functionally merged Actor and Critic network models, both
        built of fully connected layers
        -- input:
            - env_space: Size of input vector from environment(int)
            - action_space: Size of output vector for action softmax onehot
            - n_hidden (optional): Number nodes in hidden layers (int)
        -- output:
            - env: The environment vector given to the models
            - act: The softmax onehot encoded action vector
            - expected_value: The expected value from the state-action pair
    """
    def __init__(self, env_space, action_space, n_hidden=6, mode=args.mode):
        super(Net, self).__init__()
        self.action_space = action_space
        self.env_space = env_space
        self.n_hidden = n_hidden
        self.Actor()
        self.Critic()

    def Actor(self):
        self.l1 = nn.Linear(env_space, n_hidden)
        self.l2 = nn.Linear(n_hidden, n_hidden)
        self.l3 = nn.Linear(n_hidden, action_space)

    def Critic(self):
        self.evn1 = nn.Linear(env_spave, n_hidden)
        self.env2 = nn.Linear(n_hidden, n_hidden)
        self.act1 = nn.Linear(action_space, n_hidden)
        self.merged1 = nn.Linear(2*n_hidden, n_hidden)
        self.merged2 = nn.Linear(n_hidden, 1)

    def forward(self, env):
        act = self.ActorForward(env)
        expected_value = self.CriticForward(env, act)
        return env, act, expected_value

    def ActorForward(self, env):
        act = F.relu(self.l1(env))
        act = F.relu(self.l2(act))
        act = F.softmax(self.l3(act))
        return act

    def CriticForward(self, env, act):
        env = F.relu(self.env1(env))
        env = F.relu(self.env2(x))
        act = F.relu(self.act1(act))
        merged = torch.cat((env, act), 0)
        merged = F.relu(self.merged1(merged))
        expected_value = F.relu(self.merged2(merged))
        return expected_value

    def predict_action(self, state):
        """ Converts Net output Variable from a "softmax onehot encoding" into
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
        self.eval()
        possible_actions = self.forward(state).data
        action = torch.distribution.Categorical(possible_actions).sample().numpy()[0]
        #_, indicies = var.data.max(0)
        #return a max action if there are more than one
        #action = indicies[randint(0, len(indicies)-1)]
        return action

def train(actor_critic):
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
    optimizer = optim.RMSprop(policy.parameters(), args.lr, eps=args.eps, alpha=args.alpha)
    criterion = nn.MSELoss()
    actor_critic.train()
    for episode in range(0, args.num_episode):
        state = env.reset()
        frame, cum_reward, grad_policy, grad_reward = 0, 0, 0, 0
        isterminal = False
        state_store = [0 for x in range(0, args.max_frames)]
        action_store = [0 for x in range(0, args.max_frames)]
        reward_store = [0 for x in range(0, args.max_frames)]
        while (isterminal != True) or (frame < args.max_frames):
            action = predict_action(policy(state))
            state, reward, isterminal, more_info = env.step(action)
            state_store[frame] = state
            action_store[frame] = action
            reward_store[frame] = reward
            frame += 1
        if isterminal == True:
            cum_reward = expected_reward[reward] #Bootstrap from last state
        #now accumulate gradients, rewards
        while frame >= 0:
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
    parser.add_argument('--lr', type=float32, default=0.99,\
                        help='Learning rate for RMSprop optimizer')
    parser.add_argument('--alpha', type=float32, default=0.99,\
                        help='Alpha for RMSprop optimizer')
    parser.add_argument('--eps', type=float32, default=1e-5,\
                        help='Epsilon for RMSprop optimizer')
    args = parser.parse_args()

    if mode == 'train':
        #environment observation is np.array length 8
        #environment step is np.array length 4
        actor_critic = ActorCritic(env_space=8, action_space=4)
        actor_critic = train(actor_critic)
