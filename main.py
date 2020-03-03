import torch
reward = torch.randn(100)
GAMMA = .95
done = torch.randn(100)
q_stable = torch.rand((100, 33))


adder = torch.mul((GAMMA * (1-done)).unsqueeze(1).expand_as(q_stable), q_stable)

subtractor = torch.add(reward.unsqueeze(1).expand_as(adder), adder)

print(subtractor)