from model import DDPG_Metamodel
from utils import ReplayMemory
from rule_env import RuleHolder, Rule

from math import exp
from tqdm import tqdm
from random import random, randint
from itertools import count
from collections import namedtuple

import torch
import datasets
import time
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

device = 'gpu' if torch.cuda.is_available() else 'cpu'

print(f"Running on {device}")

dataset = datasets.ITA_Phonitalia()

env = RuleHolder(dataset)
test_env = RuleHolder(dataset)

# Initialize rules
'''
for environment in [env, test_env]:
    environment.add_rule('ad', 'k')
    environment.add_rule('af', 'z')
    environment.add_rule('cd', 'q')
    environment.remove_rule('abc')
'''

# Creates a named tuple constructor called transition, containing specified elements
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward',  'next_state', 'done'))

cache = ReplayMemory(1000)

BATCH_SIZE = 50
GAMMA = 0.999
RHO = 0.995
# Decay would be better served by a function inverted sigmoid or the like
#EPS_START = 0.9
#EPS_END = 0.05
#EPS_DECAY = 200
#TARGET_UPDATE = 10
MAX_EPISODES = 50
STEPS_PER_EPOCH = 50
ACT_LIMIT = 5 # sets the limit of action entry
MAX_EPISODE_LEN = 20
UPDATE_AFTER = 5
UPDATE_EVERY = 10
NUM_TEST_EPISODES = 1
START_STEPS = 5
ACTION_NOISE = 0.1

metamodel = DDPG_Metamodel(env.observation_space(), env.action_space())

def select_action(observation, noise_scale):
    "Adds noise to action and clips it within +- ACT_LIMIT"
    action = metamodel.act_unstable(observation)
    action += noise_scale * torch.randn(env.action_space())
    return np.clip(action, -ACT_LIMIT, ACT_LIMIT)

# List of all episod accuracies
episode_acc = []

def plot_accuracy():
    plt.figure(2)
    plt.clf()
    accuracy_t = torch.tensor(episode_acc, dtype=torch.float32)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Accuracy')
    plt.plot(accuracy_t.numpy())

    if len(accuracy_t) >= 100:
        means = accuracy_t.unfold(0, 100, 1).mean(1).view(-1)
        #means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    
    plt.pause(0.001)

def cache_batch(cache, batch_size):
    "From cache createsa a batch to train the Q-network"
    # unzips the Transition namnedtuples in cache to produce batches of each item in the tuple
    metabatch = Transition(*zip(*cache.sample(batch_size)))

    # concatenates in batches
    state_batch = torch.cat(metabatch.state, dim=0)
    action_batch = torch.cat(metabatch.action, dim=0)
    reward_batch = torch.cat(metabatch.reward, dim=0)
    next_state_batch = torch.cat(metabatch.next_state, dim=0)
    done_batch = torch.cat(metabatch.done, dim = 0)

    return state_batch, action_batch, reward_batch, next_state_batch, done_batch

def observation_batch(cache, batch_size):
    "gets a patch of states to train the Policy net"
    metabatch = Transition(*zip(*cache.sample(batch_size)))
    return torch.cat(metabatch.state, dim=0)


def Q_loss():
    "Comutes loss of the q network according to the revised bellman equation"
    obs, action, reward, next_obs, done = cache_batch(cache, BATCH_SIZE)

    q_unstable = metamodel.Q_unstable(obs, action)

    # Computes the subtractor in the Bellman Equation
    with torch.no_grad():
        q_stable = metamodel.Q_stable(next_obs, metamodel.Policy_stable(next_obs))
        #print(reward.shape, done.shape. q_stable.shape, sep="\n")
        #subtractor = reward + GAMMA * (1 - done) * q_stable
        adder = torch.mul((GAMMA * (1-done)).unsqueeze(1).expand_as(q_stable), q_stable)
        subtractor = torch.add(reward.unsqueeze(1).expand_as(adder), adder)

    # MSE Loss against subtractor
    return ((q_unstable - subtractor)**2 ).mean()

def Policy_loss():
    """
    Computes simple L1 norm loss on the policy network
    We use gradient descent on the negated loss to ascend the gradient
    """
    obs = observation_batch(cache, BATCH_SIZE)
    policy_loss = metamodel.Q_unstable(obs, metamodel.Policy_unstable(obs)) 
    return - policy_loss.mean()


def test_policy():
    "Tests the evironment on a complete set of actions using the stable env"
    for j in range(NUM_TEST_EPISODES):
        observation, done, episode_reward, episode_lenght = test_env.reset(), 0, 0, 0
        observation, reward, done, _ = test_env.step(select_action(observation, 0), complete=True)
        episode_reward += reward
        episode_lenght += 1
        print(f"Episode: {i_episode}\tReward: {episode_reward/len(env.dataset)}\tLenght: {episode_lenght}")

    
def optimize_model():
    "Function optimises the networks and transfers weighted parameters"
    # checks if there is enough data in the cache to begin training
    if len(cache) < BATCH_SIZE:
        return False
    
    # steps the q network
    metamodel.Q_optim.zero_grad()
    q_loss = Q_loss()
    q_loss.backward()
    metamodel.Q_optim.step()

    # turns off to not compute useless gradients
    for parameter in metamodel.Q_unstable.parameters():
        parameter.requires_grad = False

    # steps the policy network
    metamodel.Policy_optim.zero_grad()
    policy_loss = Policy_loss()
    policy_loss.backward()
    metamodel.Policy_optim.step()

    # turns back on
    for parameter in metamodel.Q_unstable.parameters():
        parameter.requires_grad = True

    # transfers parameters, weighted for rho to stabilise
    with torch.no_grad():
        for unstable_parameter, stable_parameter in zip(metamodel.Q_unstable.parameters(),
                                                        metamodel.Q_stable.parameters()):
            stable_parameter.data.mul_(RHO)
            stable_parameter.data.add_((1 - RHO)* unstable_parameter.data)

    return True


for i_episode in range(MAX_EPISODES):
    episode_reward = 0
    observation = env.reset()

    for i_step in tqdm(range(STEPS_PER_EPOCH)):

        # Does explorative actions for an ammount of steps
        if i_episode * STEPS_PER_EPOCH + i_step < START_STEPS:
            action = select_action(observation, ACTION_NOISE)
        else:
            action = torch.randn(env.action_space()) # should be implemented as actionspace BOX

        # Stepping the Environment
        obs_prime, reward, done, _ = env.step(action)
        episode_reward += reward

        if done:
            print("Got one")

        # pushes the performed action, state and reward into the cache
        cache.push(observation.unsqueeze(0),
                action.unsqueeze(0),
                reward.unsqueeze(0).float(),
                obs_prime.unsqueeze(0),
                done.unsqueeze(0).float())
        

        #Update to the most recent observation
        observation = obs_prime
        status = optimize_model()

    if status:
        test_policy()
        
print('Complete')
plt.show()