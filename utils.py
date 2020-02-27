from collections import namedtuple

Transitions = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def get_alphabet(string_roll):
    '''
    Takes a list of strings and returns the set string of all characters
    '''
    lexicon = set()
    counter = 0
    for word in string_roll:
        for l in list(word):
            if l not in lexicon:
                lexicon.add(l)
    return ''.join(lexicon)