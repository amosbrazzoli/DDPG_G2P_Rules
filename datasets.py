import pandas as pd
import torch.utils.data as d

from utils import get_alphabet

class ENG_WUsL(d.Dataset):
    '''
    Implementa a class for the English Lexicon Project Datase from the Washington University in Saint Louis
    available at: https://elexicon.wustl.edu/query13/query13.html
    '''
    def __init__(self):
        super().__init__()

        self.path = 'Datasets/ENG/WUsLData.csv'
        self.data = pd.read_csv(self.path,
                                usecols=["Word","Pron"],
                                delimiter=',',
                                na_values='#')
        self.data.Pron = self.data.Pron.replace('[""\.]','',regex=True)
        self.data.dropna(axis=0,
                        how='any',
                        inplace=True)
        self.data.reset_index(inplace=True)
        self.word_abc = get_alphabet(self.data.Word.to_list())
        self.pron_abc = get_alphabet(self.data.Pron.to_list())
        # added for naming and lexical decision

        # implements the iterator
        self.i = 0
    
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

if __name__ == "__main__":
    dataset = ENG_WUsL()
    for word, pron in dataset:
        print(word, pron)
