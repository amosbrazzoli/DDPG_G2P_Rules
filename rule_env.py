from random import sample, randint
from collections import defaultdict


def convert(word):
    conversion_dict = {'abc' : 'z',
                        'bca' : 'xl',
                        'cab' : 'xo',
                        'ab' : 'k',
                        'ca' : 'w',
                        'ba' : 'h',
                        'ac' : 'j',
                        'bc' : 'i',
                        'a' : 'a',
                        'b' : 'b',
                        'c' : 'c'}
    i = 0
    out = []
    #print(word)
    while i < len(word):
        j = 3
        while 1 <= j <= 3 :
            #print(i, i+j, word[i:i+j])
            if conversion_dict.get(word[i:i+j], None) != None:
                out.append(conversion_dict[word[i:i+j]])
                i = i+j
                break
            else:
                j -= 1
    return ''.join(out)

charset = {'a','b','c'}

def setsample(set, maxlen=False):
    if not maxlen:
        maxlen = randint(1,21)
    temp = []
    out = ''
    while len(out) < maxlen:
        temp.append(out)
        temp.append(''.join(sample(set, min(3, maxlen-len(out)))))
        out = ''.join(temp)
        temp = []
    return out

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
        self.rules = {}
        if init_dict:
            for k, v in init_dict.items():
                self.add_rule(k, v)
        self.target_dict = {}
        for i, label in enumerate(self.rules.keys()):
            self.target_dict[i] = label


    def add_rule(self, k, v, w=0):
        if self.rules.get(k, None) == None:
            self.rules[k] = Rule(k, v, w)
        else:
            raise KeyError(f'Key {k} is already set as: {self.rules[k]}')

    def read(self, word):
        out = []
        pos_dict = defaultdict(lambda: list())
        i = 0
        while  i <len(word):
            j = self.maxlen
            while 1 <= j <= self.maxlen:
                #print(i, i+j, word[i:i+j])
                if self.rules.get(word[i:i+j], None) != None:
                    for index, phone, weight in self.rules[word[i:i+j]].read(i):
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

    def action(self, target, weight):
        self.rules[self.target_dict[target]].action(weight)

    def __repr__(self):
        row_list = []
        for k, value in self.rules.items():
            row_list.append(f"{k}->{value.__repr__()}")
        return "\n".join(row_list)




'''
ruldict = {'abc' : 'z',
            'bca' : 'xl',
            'cab' : 'xo',
            'ab' : 'k',
            'ca' : 'w',
            'ba' : 'h',
            'ac' : 'j',
            'bc' : 'i',
            'a' : 'a',
            'b' : 'b',
            'c' : 'c'}

r = RuleHolder(3, ruldict)
r.action(3, 1)
print(r)

words = [setsample(charset) for _ in range(100)]
prons = list(map(convert, words))

for word, pron in zip(words, prons):
    if r.read(word) != pron:
        print('_______________________')
        print(word, r.read(word), pron)
'''
