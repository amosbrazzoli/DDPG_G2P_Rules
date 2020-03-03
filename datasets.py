import pandas as pd
import torch.utils.data as d

from random import randint, sample
from utils import get_alphabet

class Dataset:
    def __init__(self, data, headers):
        self.header = headers
        self.data = data
        self.words_name = headers[0]
        self.prons_name = headers[1]
        self.i = 0
        self.word_abc = get_alphabet(self.data[self.words_name].to_list())
        self.pron_abc = get_alphabet(self.data[self.prons_name].to_list())
        self.sampled_indexes = set()

    def __getitem__(self, index):
        return self.data[self.words_name].iloc[index].lower(), self.data[self.prons_name].iloc[index]

    def __next__(self):
        if self.i < self.__len__():
            self.i +=1
            return self[self.i-1]
        else:
            raise StopIteration()

    def __iter__(self):
        return self

    def __len__(self):
        return self.data.shape[0]

    def reset(self):
        self.i = 0

    def sample(self, sample_lenght):
        while len(self.sampled_indexes) < sample_lenght:
            while True:
                x = randint(0, self.__len__()-2) # -2 to handle for heathers and last included
                if x not in self.sampled_indexes:
                    self.sampled_indexes.add(x)
                    try:
                        yield self[x]
                    except:
                        print(x, self[x-1])
                    break
        self.sampled_indexes = set()

class ENG_WUsL(Dataset):
    def __init__(self, path='Datasets/ENG/WUsLData.csv', headers=["Word","Pron"]):
        self.path = path
        self.headers = headers
        self.data = pd.read_csv(self.path,
                            usecols=self.headers,
                            delimiter=',',
                            na_values='#')
        self.data.Pron = self.data[self.headers[1]].replace(r'[""\.]','',regex=True)
        self.data.dropna(axis=0,
                    how='any',
                    inplace=True)
        self.data.reset_index(inplace=True)
        super().__init__(self.data, self.headers)

class ITA_Phonitalia(Dataset):
    def __init__(self, path="Datasets/ITA/Phonitalia.csv", headers=["Word", "Pron"]):
        self.path = path
        self.headers = headers
        self.data = pd.read_csv(self.path,
                                    delimiter=",")
        self.data.dropna(inplace=True)
        self.default_rules = [("a", "a"),
                                ("b", "b"),
                                ("c" , "k"),
                                ("d", "d"),
                                ("e", "e"),
                                ("e", "E"),
                                ("f", "f"),
                                ("g", "G"),
                                ("gh", "G"),
                                ("h", ""),
                                ("i", "i"),
                                ("l", "l"),
                                ("m", "m"),
                                ("n", "n"),
                                ("o", "O"),
                                ("o", "o"),
                                ("p", "p"),
                                ("qu", "kw"),
                                ("r", "r"),
                                ("s", "s"),
                                ("t", "t"),
                                ("u", "u"),
                                ("v", "v"),
                                ("z", "z"),
                                ("x", "ks"),
                                ("y", "i"),
                                ("j", "j"),
                                ("gl", "LL"),
                                ("sc", "SS"),
                                ("gn", "NN"),
                                ("k", "k"),
                                ("w","w"),
                                ("q", "k")]
        super().__init__(self.data, self.headers)

class Dummy:
    def __init__(self, lenght=10_000):
        self.lenght = lenght
        self.default_rules = [('abc', 'zh'),
                            ('bca', 'xl'),
                            ('cab', 'xo'),
                            ('ab', 'k'),
                            ('ca', 'w'),
                            ('ba', 'h'),
                            ('ac', 'j'),
                            ('bc', 'i'),
                            ('a', 'a'),
                            ('b', 'b'),
                            ('c', 'c')]

        self.default_rules_dic = {k : value for (k, value) in self.default_rules }
 
        self.rule_max_len = max(map(lambda x: len(x[0]), #maps the lenght of the first element
                                    self.default_rules))
        self.charset = {'a', 'b', 'c'}
        self.words = [self.setsample() for _ in range(lenght)]
        self.prons = list(map(self.convert, self.words))

        self.i = 0

    def __getitem__(self, index):
        return self.words[index], self.prons[index]

    def __next__(self):
        if self.i < self.lenght:
            self.i +=1
            return self[self.i-1]
        else:
            raise StopIteration()
    
    def __iter__(self):
        return self

    def __len__(self):
        return self.lenght

    def reset(self):
        self.i = 0
    
    def setsample(self, maxlen=False):
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
            temp.append(''.join(sample(self.charset,
                                        min(self.rule_max_len,maxlen-len(out)))))
            out = ''.join(temp)
            temp = []
        yield out
    
    def convert(self, word):
        '''
        Converts a word using a convertion dicitionary
        uses the longest aplicable rule first
        '''
        i = 0
        out = []
        while i < len(word):
            j = self.rule_max_len
            while 1 <= j <= self.rule_max_len:
                #print(i, i+j, word[i:i+j])
                if self.default_rules_dic.get(word[i:i+j], None) != None:
                    out.append(self.default_rules_dic[word[i:i+j]])
                    i = i+j
                    break
                else:
                    j -= 1
        return ''.join(out)
        
if __name__ == "__main__":
    dataset = ITA_Phonitalia()