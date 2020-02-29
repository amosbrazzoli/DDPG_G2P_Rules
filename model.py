import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils import prod

class DQN(nn.Module):
    def __init__(self, observations_shape, action_shape):
        super(DQN, self).__init__()
        self.linearised = prod(observations_shape) + prod(action_shape)

        self.lin1 = nn.Linear(self.linearised , self.linearised//3)
        self.lin2 = nn.Linear(self.linearised//3, self.linearised//2)
        self.lin3 = nn.Linear(self.linearised//2, prod(action_shape))

    def forward(self, observation, action):
        if observation.shape[0] == action.shape[0]:
            batch_size = observation.shape[0]
        else:
            raise Exception(f'Expecting batch lenght to be \
                    {observation.shape[0]} but got {action.shape[0]} instead')

        observation = observation.view(batch_size, -1)
        action = action.view(batch_size, -1)

        ob_action = torch.cat((observation, action), dim=1)

        ob_action = F.relu(self.lin1(ob_action))
        ob_action = F.relu(self.lin2(ob_action))
        return F.relu(self.lin3(ob_action))

# Don't have to be equal, neither different
# Separation is just a clarity issue

class PolicyNet(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super(PolicyNet, self).__init__()

        observation_shape = prod(observation_shape)

        self.lin1 = nn.Linear(observation_shape, observation_shape//3)
        self.lin2 = nn.Linear(observation_shape//3, observation_shape//2)
        self.lin3 = nn.Linear(observation_shape//2, prod(action_shape))

    def forward(self, observation):
        observation = F.relu(self.lin1(observation))
        observation = F.relu(self.lin2(observation))
        return F.relu(self.lin3(observation))


class DDPG_Metamodel:
    def __init__(self,
                    observation_shape,
                    actions_shape,
                    q_lr=.001,
                    p_lr=.001,
                    optim=optim.Adam):

        self.Q_stable = DQN(observation_shape, actions_shape)
        self.Q_unstable = DQN(observation_shape, actions_shape)
        self.Q_stable.load_state_dict(self.Q_unstable.state_dict())
        self.Q_stable.eval()

        self.Policy_stable = PolicyNet(observation_shape, actions_shape)
        self.Policy_unstable = PolicyNet(observation_shape, actions_shape)

        self.Policy_stable.load_state_dict(self.Policy_unstable.state_dict())
        self.Policy_stable.eval()

        self.Q_optim = optim(self.Q_unstable.parameters(), lr=q_lr)
        self.Policy_optim = optim(self.Policy_unstable.parameters(), lr=p_lr)

    def act_unstable(self, observation):
        with torch.no_grad():
            return self.Policy_unstable(observation)

    def act_stable(self, observation):
        with torch.no_grad():
            return self.Policy_stable(observation)