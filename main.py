from model import DQN
from utils import ReplayMemory
from rule_env import RuleHolder, Rule

from math import exp
from random import random
from itertools import count

import torch
import torch.optim as optim

import matplotlib.pyplot as plt


env = RuleHolder()
# Initialize rules

Transitions = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

cache = ReplayMemory(1000)

BATCH_SIZE = 100
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


init_state = env.pull_weights()
state_size = len(init_state)

policy_net = DQN(state_size)
target_net = DQN(state_size)

# Loads the state of the policy into the target
target_net.load_state_dict(policy_net.state_dict())

# Disables learning on the target_net
target_net.eval()

# Initializes the optimizer
optimizer = optim.RMSprop(policy_net.parameters())

steps_done = 0

def select_action(state):
    global steps_done
    action_candidate = random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if action_canditate > eps_threshold:
        with torch.no_grad():
            # should return weight and index
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).view(1)
    else:
        return(torch.tensor)

episode_acc = []

def plot_accuracy():
    plt.figure(2)
    plt.clf()
    accuracy_t = torch.tensor(episode_acc, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Accuracy')
    plt.plot(accuracy_t.numpy())

    if len(accuracy_t) >= 100:
        means = accuracy_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    
    plt.pause(0.001)
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def optimize_model():
    if len(cache) < BATCH_SIZE:
        return
    transitions = cache.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

num_episodes = 50

for i_episode in range(num_episodes):

    env.reset()

    last_screen = env.pull_weights()
    current_screen = env.pull_weights()
    state = current_screen - last_screen

    for t in count():
        action = select_action(state)
        _, reward, done, _ = env.step(action.item())


