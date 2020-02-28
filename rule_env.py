from random import sample, randint, random
from collections import defaultdict, OrderedDict


class Rule:
    def __init__(self, graph, phone, w = False):
        self.graph = graph
        self.phone = phone
        if not weight:
            self.weight = random()

    def __repr__(self):
        return f"R:({self.graph},{self.weight},{self.phone})"
    
    def __len__(self):
        return len(self.graph), len(self.phone)

    def action(self, weight):
        self.weight = weight

    def read(self, i):
        for j in range(len(self.phone)):
            yield i+j, self.phone[j], self.weight

    def reset(self):
        self.weight = random()


class RuleHolder:
    def __init__(self, Dataset, maxlen=False, init_dict=False):
        self.dataset = Dataset
        self.rules = []
        self.target_dict = {}
        self.macrocounter = 0
        self.maxcount = 500
        if init_dict:
            for k, v in init_dict.items():
                self.add_rule(k, v)
        if not maxlen:
            temp_max_len = 0
            for rule in self.rules:
                if len(rule)[0] > temp_max_len:
                    temp_max_len = len(rule)[0]
            self.maxlen = temp_max_len
        else:
            self.maxlen = maxlen

    def __repr__(self):
        row_list = []
        for rule in self.rules:
            row_list.append(f"{rule.graph} -- {rule.weight} -> {rule.phone}")
        return "\n".join(row_list)

    def __len__(self):
        if len(self.rules) == len(self.target_dict):
            return len(self.rules)

    def add_rule(self, k, v, w=0):
        if self.target_dict.get(k, None) == None:
            self.target_dict[k] = len(self.rules)
            self.rules.append(Rule(k, v, w))
        else:
            raise KeyError(f'Key {k} is already set as: {self.rules[k]}')

    def _pull_weights(self):
        for rule in self.rules:
            yield rule.weight

    def set_weights(self, *args):
        if len(args) == len(self.rules):
            for i, w in enumerate(args):
                self.rules[i].weight = w
    
    def action(self, index, weight):
        self.rules[index].weight = weight
    
    def pull_weights(self):
        return list(self._pull_weights())
    

    def read(self, word):
        out = []
        pos_dict = defaultdict(lambda: list())
        i = 0
        while  i <len(word):
            j = self.maxlen
            while 1 <= j <= self.maxlen:
                #print(i, i+j, word[i:i+j])
                if self.target_dict.get(word[i:i+j], None) != None:
                    for index, phone, weight in \
                        self.rules[self.target_dict[word[i:i+j]]].read(i):
                        pos_dict[index].append((weight, phone))
                    i = i+j
                    break
                else:
                    j -= 1
        #print(pos_dict)
        for index, values in pos_dict.items():
            #print(index, values)
            out.append(sorted(values)[0][1])
        return ''.join(out)

    def reset(self):
        for rule in self.rules:
            rule.reset()
        self.macrocounter = 0

    def step(self, index, weight):
        '''
        has to return:
        observation := state
        reward := amount for the previous action
        done := if the episode has terminated
        info := diagnostic material

        Index is arg
        '''
        self.action(index, weight)
        observation = self.pull_weights

        reward = 0
        counter = 0
        for word, pron in self.dataset:
            counter += 1
            if self.read(word) == pron:
                reward +=1
        
        if reward / counter > .95:
            done = True
        elif self.macrocounter >= self.maxcount:
            done = True
        else: done = False
        self.macrocounter +=1

        return observation, reward, done, None


        
            
