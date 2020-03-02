import pandas as pd
import torch.utils.data as d

from random import randint, sample
from utils import get_alphabet

class ENG_WUsL:
    '''
    Implementa a class for the English Lexicon Project Datase from the Washington University in Saint Louis
    available at: https://elexicon.wustl.edu/query13/query13.html
    '''
    def __init__(self):
        self.path = 'Datasets/ENG/WUsLData.csv'
        self.data = pd.read_csv(self.path,
                                usecols=["Word","Pron"],
                                delimiter=',',
                                na_values='#')
        self.data.Pron = self.data.Pron.replace(r'[""\.]','',regex=True)
        self.data.dropna(axis=0,
                        how='any',
                        inplace=True)
        self.data.reset_index(inplace=True)
        self.word_abc = get_alphabet(self.data.Word.to_list())
        self.pron_abc = get_alphabet(self.data.Pron.to_list())
        # added for naming and lexical decision

        # implements the iterator
        self.i = 0

        ## FURTHER, Might implement a first way of adding rules
        self.default_rules = {}
    
    def __getitem__(self, index):
        '''
        Returns the coupling of word and pronuntiation
        '''
        x = self.data.Word.iloc[index].lower()
        y = self.data.Pron.iloc[index]
        return x, y

    def __next__(self):
        if self.i <  self.__len__():
            self.i += 1
            return self[self.i-1]
        else:
            raise StopIteration()

    def __iter__(self):
        return self

    def __len__(self):
        'Returns the lenght of the dataset'
        return self.data.shape[0]

    def reset(self):
        self.i = 0


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
        return out
    
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
        

class ITA_Phonitalia:
    def __init__(self, path="Datasets/ITA/Phonitalia.csv"):
        self.path = path
        self.data = pd.read_csv(self.path,
                                    delimiter=",")
        self.data.dropna(inplace=True)

        self.word_abc = get_alphabet(self.data.Word.to_list())
        self.pron_abc = get_alphabet(self.data.Pron.to_list())
        self.i = 0
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

    def __getitem__(self, index):
        x = self.data.Word.iloc[index]
        y = self.data.Pron.iloc[index]
        return x, y

    def __next__(self):
        if self.i < self.__len__():
            self.i += 1
            return self[self.i-1]
        else:
            raise StopIteration()

    def __iter__(self):
        return self

    def __len__(self):
        return self.data.shape[0]

    def reset(self):
        self.i = 0

        
if __name__ == "__main__":
    dataset = Dummy(10)
    print(dataset.default_rules)