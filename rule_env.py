from random import sample, randint
from collections import defaultdict, OrderedDict

class Rule:
    def __init__(self, graph, phone, w = 0):
        self.graph = graph
        self.phone = phone
        self.weight = w

    def action(self, weight):
        self.weight = weight

    def read(self, i):
        for j in range(len(self.phone)):
            yield i+j, self.phone[j], self.weight

    def __repr__(self):
        return f"R:({self.graph},{self.weight},{self.phone})"



class RuleHolder:
    def __init__(self, maxlen, init_dict=False):
        self.maxlen = maxlen
        self.rules = []
        self.target_dict = {}
        if init_dict:
            for k, v in init_dict.items():
                self.add_rule(k, v)

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

    def __repr__(self):
        row_list = []
        for rule in self.rules:
            row_list.append(f"{rule.graph} -- {rule.weight} -> {rule.phone}")
        return "\n".join(row_list)