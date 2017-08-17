"""
skmultilearn implemented classifier chains but not classifier chain ensembles. scikitlearn has ensembles of classifier 
chains, but only in the dev version. So, I wrote my own ensembling class for classifier chains. It's built on top of skmultilearn.problem_transform ClassifierChain class that creates one chain in the same order as the order of the labels
presented to it.

Basic algorithm is to:
- randomly reorder the labels and build a ClassifierChain for that ordering.
- Repeat multiple times
- Use a bagged approach with equal weighting for each ClassifierChain to predict each label and to average the
probabilities.
"""

from skmultilearn.problem_transform import ClassifierChain
import numpy as np


class ClassifierChainEnsemble(object):
    """An ensemble of ClassifierChains. ClassifierChain has been implemented in skmultilearn.
            ClassifierChainEnsemble is a meta estimator that creates an ensemble of classifier chains
            on multilabel classification problems and averages the predictions of classifiers in the ensemble to
            improve predictive accuracy.
        The order of the labels in each ClassifierChain are chosen randomly.
        I have not implemented multiclass ensemble yet. Currently, ClassifierChainEnsemble class supports binary class,
            multilabel ensembling

    Parameters
    ----------
    base_estimator : object, optional (default=None)
        The base estimator from which the ensemble is built.
    n_estimators : integer
        The number of estimators in the ensemble.
    estimator_params : list of strings
        The list of attributes to use as parameters when instantiating a
        new base estimator. If none are given, default parameters are used.

    Attributes
    ----------

    _ensemble : list of estimators
        The collection of fitted classifier chains.
    perms_list : list of permutations
        list of permutations of labels used to train classifier chains
    """


    def __init__(self, base_estimator, n_estimators=10,
                 estimator_params=tuple()):

        # Set parameters

        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimator_params = estimator_params
        # define perms_list to keep track of permutations shown to ClassifierChain
        self.perms_list = [[] for _ in range(n_estimators)]
        self._ensemble = [ClassifierChain(base_estimator) for _ in range(n_estimators)]


    def _bag(self, table):
        """
        :type table: numpy array
        _bag is an internal method, not to be called explicitly.
        """
        avg_pred = table.sum(0) / float(self.n_estimators)
        return avg_pred


    def fit(self, X, Y):
        """
        :param X: training set as numpy array of shape [n_samples, n_features]
        :param Y: training labels as numpy array of shape [n_samples, n_labels]
        :return: None
        The method trains n_estimator classifier chains, each on a random permutation of labels
        """

        # will have to implement selecting random subsets of X later
        for i in range(0, self.n_estimators):
            self.perms_list[i] = np.random.permutation(Y.shape[1])
            print self.perms_list[i]
            print Y[:, self.perms_list[i]]
            self._ensemble[i].fit(X, Y[:, self.perms_list[i]])


    def predict(self, X, rule="majority_vote"):
        """
        :param X: testing set as numpy array of shape [n_samples, n_features]
        :param rule: the polling rule used to decide between the predicted classes. Only "majority_vote" for binary classes
                     has been implemented at this point.
        :return: predicted labels for each row of X as numpy array of shape [n_test_samples, n_labels]

        ClassifierChain predicts labels in the order of labels shown to it. We undo these permutations used to train
            classifier chains (using np.argsort in the following) when predicting the labels in the test set
        """

        prediction_mat = []
        for i in range(self.n_estimators):
            reverse_perm = np.argsort(self.perms_list[i])
            prediction_mat.append(self._ensemble[i].predict(X).todense()[:, reverse_perm])
        prediction_mat = np.asarray(prediction_mat)
        result = self._bag(prediction_mat)
        if rule == "majority_vote":
            threshold = 0.5
            result[result >= threshold] = 1
            result[result < threshold] = 0
        return result


    def predict_proba(self, X):
        """
        :param X: testing set as numpy array of shape [n_samples, n_features]
        :return: predicted probability of being in one class for each row of X as numpy array of shape
            [n_test_samples, n_labels]

        ClassifierChain predicts labels in the order of labels shown to it. We undo these permutations used to train
            classifier chains (using np.argsort in the following) when predicting the labels in the test set
        """

        predict_proba_mat = []
        for i in range(self.n_estimators):
            reverse_perm = np.argsort(self.perms_list[i])
            predict_proba_mat.append(self._ensemble[i].predict_proba(X).todense()[:, reverse_perm])
        predict_proba_mat = np.asarray(predict_proba_mat)
        result = self._bag(predict_proba_mat)
        return result


if __name__ == "__main__":
    ensemble = ClassifierChainEnsemble(LogisticRegression(penalty="l1",C=1))

    """
    An example of how to use the ClassifierChainEnsemble class. Note that the synthetic dataset generated below can give errors
    that have to do with data generation rather than class implementation. In case of errors, run the script again.
    """
    # this will generate a dataset
    from sklearn.datasets import make_multilabel_classification
    x, y = make_multilabel_classification(sparse=True, n_labels=5,n_classes=3,
                                          n_samples=5,
                                          #return_indicator='dense',
                                          allow_unlabeled=False)
    #fit the dataset
    ensemble.fit(x, y)
    pred_mat = ensemble.predict(x) #predicting (on the training data)
    pred_proba_mat = ensemble.predict_proba(x)

    #print the accuracy of the ensemble for each label
    for i in range(0,y.shape[1]):
	    print sum(pred_mat[:,i]==y[:,i])/float(y.shape[0])

