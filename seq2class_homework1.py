import sys
import numpy as np
from collections import namedtuple
import csv

# Copy over the base classes that you developed during homework 1.
# Below is the same starter code that you were given for homework 1

Data_type = namedtuple('Data', ['xx', 'oo', 'yy'])

# copied from homework 1
def iterate_data(filename='train', *, max_examples=None):
    file = open(filename+'.tsv')
    for n, row in enumerate(csv.DictReader(file, delimiter='\t')):
        if max_examples and n >= max_examples:
            break
        yield Data_type(
            xx=tuple(row['xx'].split()),
            oo=tuple(row['oo'].split()) if 'oo' in row else None,
            yy=tuple(row['yy'].split()) if 'yy' in row else None,
        )

class TaskSetting(object):
    """
    Base interface for specifying a task.
    Defines the output space, the action space, and the reward function.
    """

    def iterate_yy(self, *, xx, oo=None):
        """
        Returns an iterator over legal `yy` sequences (represented as tuples).
        If an observation `oo` is specified, restricts to `yy` sequences that
        are consistent with `oo`.
        This method *defines* the space of output strings that we will consider
        (although some of those could turn out to have probability 0).
        It also *defines* how `oo` is to be interpreted.
        The default implementation generates each `yy` one character at a time,
        by calling `iterate_y`.

        Caveat to users: Some implementations (for efficiency), instead of yielding
        a stream of immutable tuples, might keep yielding mutated versions of
        the *same* list object.  Thus, you should print, analyze,
        or copy the `yy` values as you iterate through them.  Don't write `list(iterator)`
        since that might give a list of many pointers to the *same* object; to save all
        values in a list, you should instead write `[tuple() for yy in iterator]`.
        """
        ### STUDENTS START
        raise NotImplementedError()  # REPLACE ME
        ### STUDENTS END

    def iterate_y(self, *, xx, oo=None, yy_prefix):
        """
        Returns an iterator over legal next characters of `yy`.
        In other words, returns all characters `y` such that
        the concatenation `y_prefix + y` is a prefix of some
        output `yy` that is legal given input string `xx`
        and observable `oo`.
        (This set of characters is discussed in formalisms.pdf,
        "Restricting summations to the output space".)

        How do we know when we have reached the end of `yy`?
        For this course, you can just assume `len(yy)=len(xx)`.
        However, a more general design is to use `None` as an EOS
        symbol.  If you prefer that design, then `iterate_y`
        should also yield `None` if `y_prefix` can itself serve
        as a legal output `yy`.
        """
        raise NotImplementedError()  # implement this below

    def iterate_aa(self, *, xx):
        """
        Returns an iterator over plans that are allowed for input `xx`.
        The default implementation just calls `iterate_yy(xx=xx)`, which is
        appropriate for prediction tasks where the plans simply correspond
        to predicting the different outputs.  This can be overridden for
        other kinds of decision tasks.
        (See formalisms.pdf, "Decision theory" and "More decision theory".)
        """
        yield from self.iterate_yy(xx=xx)

    def reward(self, *, aa, xx, yy):
        """
        Return the reward that plan `aa` will get on input `xx` if the true answer is `yy`.
        This method *defines* the reward function.
        """
        assert yy is not None    # should appear in subclass implementations too
        raise NotImplementedError()

    def reward_threshold(self, *, xx):
        """
        Return a value `t` such that we consider plans with reward >= `t` to be "good"
        and plans with reward < `t` to be "errors".  This can be used for listing errors
        and serves as additional documentation of the reward function.  Note that the
        threshold may depend on `xx`.
        """
        raise NotImplementedError()


class ProbabilityModel(object):
    """
    Base class for conditional probability models P_theta(y | x).
    """

    def __init__(self, task):
        assert isinstance(task, TaskSetting)
        self.task = task
        self.initialize_params()

    def initialize_params(self):
        """
        Reset the model parameters to their start state.
        """
        raise NotImplementedError()

    def prob(self, *, xx, oo=None, yy=None):
        """
        Return p(yy | xx) or p(oo | xx).  Only one of `yy` or `oo` should be specified.

        If `yy` is not a legal string in self.task's output space, or `oo` is not a
        legal observable, then we would ideally raise an error, but you are not
        required to implement that.
        """
        assert (oo is None) != (yy is None)   # should appear in subclass implementations too
        raise NotImplementedError()

    def uprob(self, *, xx, oo=None, yy=None):
        """
        Just like `prob`, except that this version is free to return an
        unnormalized probability when that is more efficient.
        The default implementation just calls `prob`.
        """
        return self.prob(xx=xx, oo=oo, yy=yy)

    def logprob_gradient(self, *, xx, oo=None, yy=None, use_efficient_gradient=False):
        """
        The gradient of log p(yy | xx) or log p(oo | xx) with respect to the
        model parameters `params`, evaluated at the current model parameters.

        Either `yy` as a fully observed output or `oo` as a partial observation
        should be specified, but not both. (See formalisms.pdf, "Observations".)

        This is typically used to help estimate the parameters of the model.
        """
        assert (oo is None) != (yy is None)   # should appear in subclass implementations too
        raise NotImplementedError()

    def logprob_per_example(self, dataset):
        """
        Return the log of the conditional probability assigned
        by the model to an example, averaged over all examples
        in the given dataset.  This is useful to check the predictive
        power of the model `p(yy | xx)`.

        For each example, this method will sum `log p(yy | xx)` if
        `yy` is defined, and otherwise `log p(oo | xx)`.  It never
        conditions on `oo`, since the model is intended to capture
        `p(yy | xx)`.

        On a training dataset, this measures log-likelihood (how well
        the model fits the training examples).  On a dev or test dataset
        it measures held-out log-likelihood (how well the model predicts
        held-out examples).

        The default implementation calls `prob`.
        """
        total_logprob = 0
        count = 0
        for xx, oo, yy in dataset:
            total_logprob += np.log(self.prob(xx=xx, oo=oo) if yy is None
                                    else self.prob(xx=xx, yy=yy))
            count += 1
            if count % 50 == 0: sys.stdout.write('\r\tevaluated log-probability on {} examples'.format(count))
        sys.stdout.write('\n')
        return total_logprob / count

    def sampler(self, *, xx, oo=None, temperature=1):
        """
        Generates an infinite stream of samples `yy` drawn exactly
        from p(`yy` | `xx`) or p(`yy` | `xx`, `oo`).
        The default method uses brute force via `self.task.iterate_yy`.
        """
        # Note: We return a stream to avoid redoing work when we want *many* samples.
        # You should only have to compute the unnormalized probabilities and the
        # normalizer once and then keep reusing them for the whole stream.

        ### STUDENTS START
        raise NotImplementedError()  # REPLACE ME
        ### STUDENTS END

    def approx_sampler(self, *, xx, oo=None, temperature=1):
        """
        An approximate version of `sampler`.
        The default method uses Gibbs sampling, so successive samples in the
        stream will be correlated.

        Caveat to users: Same caveat as at TaskSampler.iterate_yy.
        """
        ### STUDENTS START
        raise NotImplementedError()  # REPLACE ME
        ### STUDENTS END


class BoltzmannModel(ProbabilityModel):
    """
    Base class for conditional probability models of the form
    P_theta(y | x) = (1/Z(x)) exp (G_theta(x,y) / T),
    that is, a Boltzmann distribution with temperature T.
    We refer to G_theta(x,y) as a "score".
    """

    def score(self, *, xx, yy):
        """
        Return the score G(`xx`, `yy`) as defined by the current params.
        By default, call `score_with_gradient` and only return the score.
        """
        score, gradient = self.score_with_gradient(xx=xx, yy=yy)
        return score

    def score_with_gradient(self, *, xx, yy):
        """
        Return two values: the score G(`xx`,`yy`) and its gradient with respect to the params.
        It's often convenient to compute the gradient along with the score, and we'll sometimes
        need the gradient.
        This method usually *defines* G.
        """
        raise NotImplementedError()


    def normalizer(self, *, xx, oo=None, temperature=1):
        """
        The normalizing function `Z(xx)` or `Z(xx,oo)`,
        often called the "partition function", that is used to define
            p(yy | xx)     = \frac{1}{Z(xx)}    exp G(...)
            p(yy | xx, oo) = \frac{1}{Z(xx,oo)} exp G(...)
        (See formalisms.pdf, "Marginal and conditional probabilities".)

        The default implementation computes this by a brute-force sum with `iterate_yy`.
        However, that could be overridden by a more efficient method when available,
        or by an approximation.
        """
        ### STUDENTS START
        raise NotImplementedError()  # REPLACE ME
        ### STUDENTS END

    def uprob(self, *, xx, oo=None, yy=None, temperature=1):
        assert (oo is None) != (yy is None)
        ### STUDENTS START
        raise NotImplementedError()  # REPLACE ME
        ### STUDENTS END

    def prob(self, *, xx, oo=None, yy=None, temperature=1):
        assert (oo is None) != (yy is None)
        ### STUDENTS START
        raise NotImplementedError()  # REPLACE ME
        ### STUDENTS END

    def logprob_gradient(self, *, xx, oo=None, yy=None):
        assert (oo is None) != (yy is None)
        # Warning: Don't inadvertently recompute the same normalizer many times!  That's inefficient.
        ### STUDENTS START
        raise NotImplementedError()  # REPLACE ME
        ### STUDENTS END


class DecisionAgent(object):
    """
    Base class for the decision agents in this homework.

    The `decision` function in subclasses should implement some
    decision rule, which may refer to `self.task` (a `TaskSetting`)
    and `self.model` (a `ProbabilityModel`).
    """

    def __init__(self, task, model):
        """
        Arguments to the constructor are a TaskSetting
        and a ProbabilityModel.
        """
        super().__init__()
        assert isinstance(task, TaskSetting)
        assert isinstance(model, ProbabilityModel)
        self.model = model
        self.task = task

    def decision(self, *, xx, oo=None):
        """
        Return some action `aa` that is appropriate to input `xx` and the partially
        observed output `oo` (if any).

        This is the agent's decision rule.  It might make use of `model`, `reward`,
        `iterate_aa`, and/or a random number generator.
        """
        raise NotImplementedError()

    def test(self, dataset):
        """
        Run the decision rule on all the examples in `dataset`
        and return the average reward.
        """
        reward = 0
        count = 0
        for xx, oo, yy in dataset:
            aa = self.decision(xx=xx, oo=oo)
            reward += self.task.reward(aa=aa, xx=xx, yy=yy)
            count += 1
            if count % 50 == 0: sys.stdout.write('\r\tevaluated reward on {} examples'.format(count))
        sys.stdout.write('\n')
        return reward / count

    def show_errors(self, dataset, max_examples=20, reward_threshold=None):
        """
        Print (up to) `max_examples` examples in which the decision
        rule made a "bad" decision — one with reward < `reward_threshold`.
        `reward_threshold` may be a constant number or a function of the input `xx`,
        By default, it is the method task.reward_threshold.
        """
        # Modify reward_threshold if needed so that it's a function of `xx`
        if not callable(reward_threshold):
            if reward_threshold is None:
                reward_threshold = lambda xx: task.reward_threshold(xx=xx)
            else:  # it should be a constant number
                threshold = reward_threshold
                reward_threshold = lambda xx: threshold

        # Iterate over the data
        for xx, oo, yy in dataset:
            aa = self.decision(xx=xx, oo=oo)
            r = self.task.reward(aa=aa, xx=xx, yy=yy)
            if r < reward_threshold(xx):
                print(" r: {r}\n\tyy: {yy}\n\taa: {aa}\n\txx: {xx}\n\too: {oo}".format(
                     r=r,
                    xx=''.join(xx),
                    oo=''.join(oo),
                    yy=''.join(yy),
                    aa=''.join(aa),
                ))
                max_examples -= 1
                if max_examples == 0: break

    ## The following methods are discussed later in the assignment.
    ## They ensure that decision agents are trainable.

    def stochastic_gradient(self, **kwargs):
        """
        In general, a decision agent might have its own parameters, which
        it might train in such a way as to maximize reward.  (This is
        particularly important in reinforcement learning.)

        By default, however, if the agent is asked to train on an example, it
        will simply use the example to train the underlying probability model.
        """
        return self.model.stochastic_gradient(**kwargs)

    # By default, the params of the decision agent are the params of the underlying model.
    @property
    def params(self):
        return self.model.params

    @params.setter
    def params(self, val):
        self.model.params = val


class ViterbiAgent(DecisionAgent):

    def decision(self, *, xx, oo=None):
        """
        The Viterbi decision rule.
        Use brute force with iterate_yy to enumerate the entire range of Y
        ***Same as what you wrote for homework 1***
        """
        ### STUDENTS START
        raise NotImplementedError()  # REPLACE ME
        ### STUDENTS END


class BayesAgent(DecisionAgent):
    """
    Try to make the decision that minimizes the Bayes risk
    (or in positive terms, maximizes the Bayes value).
    """

    def decision(self, *, xx, oo=None):
        # Warning: Don't inadvertently recompute the same normalizer many times!  That's inefficient.
        ### STUDENTS START
        ### Bayes min risk decoding
        raise NotImplementedError()  # REPLACE ME
        ### STUDENTS END


class L2LogLikelihood:
    """
    This class can be mixed into a ProbabilityModel to
    set a training objective of maximizing its L2-regularized
    log-likelihood, or equivalently, minimizing the negation
    of that.

    `regularization_coeff` is an attribute that can be modified
    directly and can also be specified by a keyword argument to
    the constructor.  The same is true for `num_examples`.
    """

    def __init__(self, *args, regularization_coeff=1, num_examples=None, **kwargs):
        assert regularization_coeff >= 0
        assert num_examples is None or num_examples >= 0
        self.regularization_coeff = regularization_coeff
        self.num_examples = num_examples
        super().__init__(*args, **kwargs)

    def batch_objective_with_gradient(self, dataset):
        raise NotImplementedError()     # not needed for this assignment, and probably not for this course

    def stochastic_gradient(self, *args, **kwargs):
        """
        Note that num_examples must be specified for the stochastic
        gradient case, so that the regularization term can be partitioned
        among all the training examples.
        """
        return -(self.logprob_gradient(*args, **kwargs)
                 - (2 * self.regularization_coeff / self.num_examples) * self.params)

from random import shuffle
class SGDTrainer(object):
    """
    Algorithm for training any object by stochastic gradient ascent,
    starting at its current parameters.  The object must implement a
    `stochastic_gradient` method and have a `params` property.
    """
    def __init__(self, *, epochs=1, init_stepsize=0.05, decay=0):   # could add other convergence criteria
        self.epochs = epochs
        self.init_stepsize = init_stepsize
        self.decay = decay

    def train(self, trainable, dataset):
        iteration = 0              # count the number of updates so far
        dataset = list(dataset)    # make an internal copy
        print('\ttraining on dataset of {} examples'.format(len(dataset)))
        for _ in range(self.epochs):
            shuffle(dataset)       # permute in place so that the examples are visited in a random order
            for example in dataset:
                example = dict(example._asdict())   # unpack named tuple into regular tuple
                stepsize = self.init_stepsize / (1 + self.init_stepsize * self.decay * iteration)
                      # stepsize decreases slowly over time
                trainable.params -= stepsize * trainable.stochastic_gradient(**example)
                iteration = iteration+1
                if iteration % 50 == 0:
                    sys.stdout.write('\r\ttrained for {} iterations'.format(iteration))  # print progress
        sys.stdout.write('\n')
