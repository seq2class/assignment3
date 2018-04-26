import numpy as np
import sys
import math

from seq2class_homework1 import (
    TaskSetting,
    DecisionAgent
)


class IobTask0(TaskSetting):

    Y_alphabet = tuple('IOB')    # the output alphabet for this task

    def __init__(self, false_pos_penalty=1):
        super().__init__()
        self.false_pos_penalty = false_pos_penalty    # lambda

    def iterate_y(self, *, xx, oo=None, yy_prefix):
        ### STUDENTS START
        raise NotImplementedError()  # REPLACE ME
        ### STUDENTS END

    # Could just use the inherited iterate_yy method, but let's extend it with assertions
    # that the strings we're returning are actually legal.  (This might catch a bug, e.g.,
    # you forgot to check whether a character observed in oo was legal given the prefix.)
    def iterate_yy(self, xx, oo=None):
        for yy in super().iterate_yy(xx=xx,oo=oo):
            assert yy[0] != 'I'                # 'I' is illegal following BOS
            assert 'OI' not in ''.join(yy)     # 'I' is illegal following 'O'
            yield yy

    def reward(self, *, aa, xx, yy):
        """
        The proxy reward of prediction aa on this sentence if the true chunking is yy.
        """
        # Hint: call reward_F1_triple
        ### STUDENTS START
        raise NotImplementedError()  # REPLACE ME
        ### STUDENTS END

    def reward_F1_triple(self, *, aa, xx, yy):
        """
        ***NEW***: returns a triple (true_pos, true, pos) used to compute corpus-level F1:
           `true_pos` is the number of "true positives" (chunks reported by `aa` that are truly in `yy`)
           `true` is the number of chunks that are truly in `yy`
           `pos` is the number of chunks reported by `aa`
        """
        assert len(aa) == len(yy)
        true = sum('B' == y for y in yy)
        pos = sum('B' == a for a in aa)
        true_pos = 0
        I = False    # are we currently inside a true_pos chunk?
        for i in range(len(aa)):
            if I:
                if aa[i] != 'I' or yy[i] != 'I':
                    I = False       # aa and yy didn't both continue the chunk they were both in
                    if aa[i] != 'I' and yy[i] != 'I':
                        true_pos += 1   # aa and yy both ended the chunk they were both in, at same place i-1
            if aa[i] == 'B' and yy[i] == 'B':
                I = True    # both aa and yy started a new chunk at same place i
        true_pos += I       # if I==True, then aa and yy both ended the chunk they were both in at end of string
        return np.array([true, pos, true_pos])


class Integerizer(object):
    """
    A collection of distinct object types, such as a vocabulary or a set of parameter names,
    that are associated with consecutive ints starting at 0.
    """

    def __init__(self, iterable=[]):
        """
        Initialize the collection to the empty set, or to the set of *unique* objects in its argument
        (in order of first occurrence).
        """
        # Set up a pair of data structures to convert objects to ints and back again.
        self._objects = []   # list of all unique objects that have been added so far
        self._indices = {}   # maps each object to its integer position in the list
        # Add any objects that were given.
        self.update(iterable)

    def __len__(self):
        """
        Number of objects in the collection.
        """
        return len(self._objects)

    def __iter__(self):
        """
        Iterate over all the objects in the collection.
        """
        return iter(self._objects)

    def __contains__(self, obj):
        """
        Does the collection contain this object?  (Implements `in`.)
        """
        return self.index(obj) is not None

    def __getitem__(self, index):
        """
        Return the object with a given index.
        (Implements subscripting, e.g., `my_integerizer[3]`.)
        """
        return self._objects[index]

    def index(self, obj, add=False):
        """
        The integer associated with a given object, or `None` if the object is not in the collection (OOV).
        Use `add=True` to add the object if it is not present.
        """
        try:
            return self._indices[obj]
        except KeyError:
            if add:
                # add the object to both data structures
                i = len(self)
                self._objects.append(obj)
                self._indices[obj] = i
                return i
            else:
                return None

    def add(self, obj):
        """
        Add the object if it is not already in the collection.
        Similar to `set.add` (or `list.append`).
        """
        self.index(obj, add=True)  # call for side effect; ignore return value

    def update(self, iterable):
        """
        Add all the objects if they are not already in the collection.
        Similar to `set.update` (or `list.extend`).
        """
        for obj in iterable:
            self.add(obj)


def F1(true, pos, true_pos):
    """
    Compute an F1 score from the relevant count triple.
    """
    if true+pos > 0:
        f1 = 2*true_pos / (true+pos)
    else:
        f1 = 1   # full credit if there were no true chunks and we found none
    precision = (true_pos / pos) if pos > 0 else math.nan  # precision of 0/0 is not well defined
    recall = (true_pos / true) if true > 0 else 1          # recall for 0/0 is 1: full credit if there were no true chunks and we found none
    print(f'F1 {f1}, precision: {precision}, recall: {recall}, true: {true}, pos: {pos}, true_pos: {true_pos}')
    return f1


def test_F1(self, dataset):
    """
    Run the decision rule on all the examples in `dataset` and return the F1 score.
    """
    counts = np.zeros(3)  # return true, pos, true_pos
    for c, (xx, oo, yy) in enumerate(dataset):
        aa = self.decision(xx=xx, oo=oo)
        counts += self.task.reward_F1_triple(aa=aa, xx=xx, yy=yy)   # running total
        if c % 50 == 49: sys.stdout.write('\r\tevaluated reward on {} examples'.format(c+1))
    sys.stdout.write('\n')

    # compute the f1 score.  We use a rearranged form of the formula, to reduce the chances of division by 0.
    true, pos, true_pos = counts

    return F1(true, pos, true_pos)

DecisionAgent.test_F1 = test_F1    # patch DecisionAgent to add a new method
