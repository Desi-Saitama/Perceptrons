import numpy as np


class Perceptron:
    """Perceptron Classifier

     Parameters
     -----
     eta : float
        Learning Rate
     n_iter : int
        number of passes over the training Dataset
     random_state : int
        seed for random number generator for
        weight initialization

    Attributes
    ------
    w_ : 1D-array
        Weights after model fitting
    b_ : Scalar
        Bias unit after model fitting

    errors_ : list
        Number of mis-classifications in each epoch
    """

    # Initialization of class
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    # Fit method
    def fit(self, X, y):
        """Fit training data

        Parameters
        -------
        X: (array like), shape = [n_examples, n_features]
            Training Vectors, where n_example is number of examples
            and n_feature is number of features.

        y: array-like, shape = [n_examples]
            target Values

        returns self:object

        """
        rgen = np.random.RandomState(self.random_state)  # random number generator
        self.w_ = rgen.normal(loc=0.0, scale=0.01,
                              size=X.shape[1])  # initializing weight for training
        self.b_ = np.float_(0.)  # initializing bias for training
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update*xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input (z) """
        return np.dot(X, self.w_)+self.b_

    def predict(self, X):
        """return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)