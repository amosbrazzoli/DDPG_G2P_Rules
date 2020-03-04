import torch
from random import sample, randint, random
from collections import defaultdict, OrderedDict


class Rule:
    def __init__(self, graph, phone, weight = False):
        self.graph = graph
        self.phone = phone
        if not weight:
            self.weight = random()
        self.len = (len(self.graph), len(self.phone))

    def __repr__(self):
        return f"R:({self.graph},{self.weight},{self.phone})"

    def __eq__(self, other):
        if self.graph == other.graph:
            if self.phone == other.phone:
                if self.weight == other.weight:
                    return True
        return False

    def action(self, weight):
        self.weight = weight

    def read(self, i):
        for j in range(len(self.phone)):
            yield i+j, self.phone[j], self.weight
    
    def reset(self):
        self.weight = random()


class RuleHolder:
    def __init__(self, Dataset, sample_size=1000, maxlen=False):
        self.dataset = Dataset
        self.sample_size = sample_size
        self.rules = []
        self.target_dict = {}
        self.macrocounter = 0
        self.maxcount = 500

        if self.dataset.default_rules:
            for k, v in self.dataset.default_rules:
                self.add_rule(k, v)

        if not maxlen:
            maxlen = 0
            for rule in self.rules:
                if rule.len[0] > maxlen:
                    print(rule.len)
                    maxlen = rule.len[0]
        self.maxlen = maxlen

    def __repr__(self):
        row_list = []
        for rule in self.rules:
            row_list.append(f"{rule.graph} -- {rule.weight} -> {rule.phone}")
        return "\n".join(row_list)

    def __len__(self):
         return len(self.rules)

    def add_rule(self, k, v, w=0):
        rule_candidate = Rule(k, v, w)
        if self.target_dict.get(k, None) == None:
            self.target_dict[k] = [len(self.rules)]
        else:
            for index in self.target_dict[k]:
                if self.rules[index] == rule_candidate:
                    raise Exception(f"Rule {rule_candidate} already present")
            self.target_dict[k].append(len(self.rules))
        self.rules.append(rule_candidate)

    def _pull_weights(self):
        for rule in self.rules:
            yield rule.weight

    def set_weights(self, args):
        if len(args) == len(self.rules):
            for i, w in enumerate(args):
                self.rules[i].weight = w

    def perturbate_weights(self, args):
        if len(args) == len(self.rules):
            for i, w in enumerate(args):
                self.rules[i].weight += w
        else:
            raise Exception(f"Expected size {len(self.rules)} bgot {len(args)}")
    
    def pull_weights(self):
        return torch.Tensor(list(self._pull_weights()))

    def observation_space(self):
        return len(self.rules)

    def action_space(self):
        return len(self.rules)
    
    def remove_rule(self, key, value):
        rules_indexes = self.target_dict.get(key, None)
        if rules_indexes != None:
            for i, list_i in enumerate(self.target_dict[key]):
                if self.rules[self.target_dict[key][list_i]].phone == value:
                    self.target_dict[key].remove(i)
                    self.rules.pop(list_i)

    def read(self, word):
        out = []
        pos_dict = defaultdict(lambda: list())
        i = 0
        while  i < len(word):
            j = self.maxlen
            while 1 <= j <= self.maxlen:
                if self.target_dict.get(word[i:i+j], None) != None:
                    for rule_index in self.target_dict[word[i:i+j]]:
                        for index, phone, weight in self.rules[rule_index].read(i):
                            pos_dict[index].append((weight, phone))
                    i, j = i+j, self.maxlen

                else:
                    j -= 1
                
        print(pos_dict)
        for index, values in pos_dict.items():
            out.append(sorted(values)[0][1])
        return ''.join(out)

    def reset_i(self):
        self.dataset.reset()

    def reset(self):
        for rule in self.rules:
            rule.reset()
        self.macrocounter = 0
        return self.pull_weights()

    def step(self, variation, complete=False):
        '''
        has to return:
        observation := state
        reward := amount for the previous action
        done := if the episode has terminated
        info := diagnostic material

        Index is arg
        '''
        if complete:
            iterator = self.dataset
        else:
            iterator = self.dataset.sample(self.sample_size)
        
        self.perturbate_weights(variation)
        observation = self.pull_weights()
        reward = 0
        for word, pron in interator:
            if self.read(word) == pron:
                reward +=1
        self.reset_i()
        if reward / self.sample_size > .95:
            done = True
        elif self.macrocounter >= self.maxcount:
            done = True
        else: done = False
        self.macrocounter +=1

        return observation, torch.tensor(reward), torch.tensor(done), None

if __name__ == "__main__":
    import datasets
    env = RuleHolder(datasets.ITA_Phonitalia())
    print(env.read("abiura"))