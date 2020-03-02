# Dataset Class Documentation
The dataset are constructed accordint to the same template
_an overclass might be constructed in the future to improve the scalability_

## Basic Infos

Datasets are based on a resettable iterator, so that you may pass through the data as you have episodes of the training

Therefore it must have the following methods:
* `__getitem__` returns an event of items
* `__next__` returning the next element, keeping track through an index `self.i`
* `__iter__` returning self
* `__len__` returning the lenght of the dataset

Additionally it must have:
* `.reset()` sets `self.i` to zero
* also it may have a `self.default_rules` initialising the default rules for the set