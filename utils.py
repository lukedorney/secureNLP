"""Contains utility functions for different models / data processing."""
import numpy as np
from sklearn.model_selection import PredefinedSplit


def get_predefined_split(train, dev):
    """Helper function for using a predefined train-dev-test split with sklearn estimators

    :param train: training set
    :param dev: development set
    :return: list containing np arrays with the indexes of all training and dev set entries
    """
    # method return is equivalent to:
    # predefined_split = [(np.asarray([x for x in range(len(train['vecs']))]),
    #                      np.asarray([x for x in range(len(train['vecs']), len(train['vecs'])+len(dev['vecs']))]))]
    predefined_split = np.repeat([-1, 0], [len(train), len(dev)])
    predefined_split = list(PredefinedSplit(test_fold=predefined_split).split())
    return predefined_split


def str2float(vec_array):
    """ Helper function to turn an list of vectors in string form into floats

    :param vec_array: list of vectors, which are tab separated reals
    :return:
    """
    return [list(map(float, vec.split())) for vec in vec_array]
