from random import sample, randint
from collections import defaultdict

from rule_env import Rule, RuleHolder

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

charset = {'a','b','c'}

def convert(word, conversion_dict):
    '''
    Converts a word using a convertion dicitionary
    uses the longest aplicable rule first
    '''
    max_lenght_dict = max(map(len, conversion_dict.keys()))
    i = 0
    out = []
    #print(word)
    while i < len(word):
        j = max_lenght_dict
        while 1 <= j <= max_lenght_dict :
            #print(i, i+j, word[i:i+j])
            if conversion_dict.get(word[i:i+j], None) != None:
                out.append(conversion_dict[word[i:i+j]])
                i = i+j
                break
            else:
                j -= 1
    return ''.join(out)



def setsample(set, maxlen=False):
    '''
    Creates a sample string of maxlen max lenght
    form a set of substrings combining them casually
    '''
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




r = RuleHolder(3, conversion_dict)
r.action(3, 1)
print(r)

words = [setsample(charset) for _ in range(100)]
prons = list(map(lambda x : convert(x, conversion_dict), words))

for word, pron in zip(words, prons):
    if r.read(word) != pron:
        print('_______________________')
        print(word, r.read(word), pron)
