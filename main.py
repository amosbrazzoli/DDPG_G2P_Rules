from model import DDPG_Metamodel
from datasets import Dummy
from utils import ReplayMemory
from rule_env import RuleHolder, Rule

from math import exp
from random import random, randint
from itertools import count
from collections import namedtuple

import torch
import time
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt

device = 'gpu' if torch.cuda.is_available() else 'cpu'

print(f"Running on {device}")

dataset = Dummy()

env = RuleHolder(dataset)
test_env = RuleHolder(dataset)

# Initialize rules
for environment in [env, test_env]:
    environment.add_rule('ad', 'k')
    environment.add_rule('af', 'z')
    environment.add_rule('cd', 'q')
    environment.remove_rule('abc')


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward',  'next_state', 'done'))

cache = ReplayMemory(1000)

BATCH_SIZE = 100
GAMMA = 0.999
RHO = 0.995
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10
MAX_EPOCHS = 10
STEPS_PER_EPOCH = 100
ACT_LIMIT = 5
MAX_EPISODE_LEN = 20
UPDATE_AFTER = 5
UPDATE_EVERY = 10
NUM_TEST_EPISODES = 3
START_STEPS = 2
ACTION_NOISE = 0.1

metamodel = DDPG_Metamodel(env.observation_space(), env.action_space())

def select_action(observation, noise_scale):
    action = metamodel.act_unstable(observation)
    action += noise_scale * torch.randn(env.action_space())
    return np.clip(action, -ACT_LIMIT, ACT_LIMIT)

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
    metabatch = Transition(*zip(*cache.sample(batch_size)))

    state_batch = torch.cat(metabatch.state, dim=0)
    action_batch = torch.cat(metabatch.action, dim=0)
    reward_batch = torch.cat(metabatch.reward, dim=0)
    next_state_batch = torch.cat(metabatch.next_state, dim=0)
    done_batch = torch.cat(metabatch.done, dim = 0)

    return state_batch, action_batch, reward_batch, next_state_batch, done_batch

def observation_batch(cache, batch_size):
    metabatch = Transition(*zip(*cache.sample(batch_size)))
    return torch.cat(metabatch.state, dim=0)


def Q_loss():
    obs, action, reward, next_obs, done = cache_batch(cache, BATCH_SIZE)

    q_unstable = metamodel.Q_unstable(obs, action)

    # Computes the subtractor in the Bellman Equation
    with torch.no_grad():
        q_stable = metamodel.Q_stable(next_obs, metamodel.Policy_stable(next_obs))
        print(reward.shape, done.shape. q_stable.shape, sep="\n")
        subtractor = reward + GAMMA * (1 - done) * q_stable

    # MSE Loss against subtractor
    return ((q_unstable - subtractor)**2 ).mean()

def Policy_loss():
    obs = observation_batch(cache, BATCH_SIZE)
    policy_loss = metamodel.Q_unstable(obs, metamodel.Policy_unstable()) 
    return - policy_loss.mean()


def test_policy():
    for j in range(NUM_TEST_EPISODES):
        observation, done, episode_reward, episode_lenght = test_env.reset(), 0, 0, 0
        while not ( done or (episode_lenght == MAX_EPISODE_LEN)):
            observation, reward, done, _ = test_env.step(select_action(observation, 0))
            episode_reward += reward
            episode_lenght += 1
        print(f"Episode: {j}\tReward: {episode_reward}\tLenght: {episode_lenght}")

    
def optimize_model():
    if len(cache) < BATCH_SIZE:
        return
    metamodel.Q_optim.zero_grad()
    q_loss = Q_loss()
    print(q_loss)
    q_loss.backward()
    metamodel.Q_optim.step()

    for parameter in metamodel.Q_unstable.parameters():
        parameter.requires_grad = False

    metamodel.Policy_optim.zero_grad()
    policy_loss = Policy_loss()
    print(policy_loss)
    policy_loss.backward()
    metamodel.Policy_optim.step()

    for parameter in metamodel.Q_unstable.parameters():
        parameter.requires_grad = True

    with torch.no_grad():
        for unstable_parameter, stable_parameter in zip(metamodel.Q_unstable.parameters(),
                                                        metamodel.Q_stable.parameters()):
            stable_parameter.data.mul_(RHO)
            stable_parameter.data.add_((1 - RHO)* unstable_parameter.data)


total_steps = MAX_EPOCHS * STEPS_PER_EPOCH
start_time = time.time()
observation, episode_reward, episode_lenght = env.reset(), 0, 0

for i_episode in range(total_steps):

    if i_episode > START_STEPS:
        action = select_action(observation, ACTION_NOISE)
    else:
        action = torch.randn(env.action_space()) # should be implemented as actionspace BOX

    # Stepping the Environment
    obs_prime, reward, done, _ = env.step(action)

    episode_reward += reward
    episode_lenght += 1

    # Ignore the "done" signal if it comes from hitting the time
    # horizon (that is, when it's an artificial terminal signal
    # that isn't based on the agent's state)
    done = torch.tensor(0) if i_episode==MAX_EPISODE_LEN else done

    #print(observation, reward, obs_prime, done, sep="\n")

    cache.push(observation.unsqueeze(0),
            action.unsqueeze(0),
            reward.unsqueeze(0).float(),
            obs_prime.unsqueeze(0),
            done.unsqueeze(0).float())

    #Update to the most recent observation
    observation = obs_prime

    #Handle end of trajectory
    if done or (episode_lenght == MAX_EPISODE_LEN):
        episode_acc.append((reward / dataset.lenght)*100 )
        plot_accuracy()
        print(f"Episode: {i_episode % STEPS_PER_EPOCH}\tReward: {episode_reward}\tLenght: {episode_lenght}")
        observation, episode_reward, episode_lenght = env.reset(), 0, 0


    #Update hanling
    if i_episode >= UPDATE_AFTER and i_episode % UPDATE_EVERY == 0:
        for _ in range(UPDATE_EVERY):
            optimize_model()
            continue

    if (i_episode + 1) % STEPS_PER_EPOCH == 0:
        epoch = (i_episode + 1) // STEPS_PER_EPOCH

        test_policy()
    
print('Complete')
plt.show()