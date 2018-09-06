"""Contains utility functions for different models / data processing."""
from msgpack import pack, unpack
import numpy as np
import os
import re
from sklearn.model_selection import PredefinedSplit


def check_dir(name):
    """Checks if a directory exists, if not then it gets created

    :param name: name of the directory to be checked
    """
    if not os.path.isdir(name):
        os.mkdir(name)


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


def prevent_sentence_boundary_detection(doc):
    """Function to keep sent boundaries found in data for spacy model.

    :param doc: spacy doc made up of processed tokens
    """
    for token in doc:
        token.is_sent_start = False
    return doc


def read_msgpack(file):
    """ Helper function for reading compressed msgpack files.

    :param file: name of the source file to be unpacked
    :return:
    """
    with open(file, 'rb') as file:
        obj = unpack(file)
    return obj


def relax(labels):
    """ Realaaax and take the B- and I- prefixes off of all labels in a portion of the data set

    :param labels: list of labels for a part of the data set
    :return: a list of labels sans B- and I-
    """
    return [[re.sub('[BI]-', '', label) for label in sentence] for sentence in labels]


def str2float(vec_array):
    """ Helper function to turn an list of vectors in string form into floats

    :param vec_array: list of vectors, which are tab separated reals
    :return:
    """
    return [list(map(float, vec.split())) for vec in vec_array]


def write_msgpack(obj, file):
    """Helper function for writing an object to file with msgpack

    :param obj: the object being unpacked with msgpack
    :param file: name of the destination file
    :return:
    """
    with open(file, 'wb') as file:
        pack(obj, file)
