from itertools import chain
import re
import time

import msgpack
import scipy
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
import sklearn_crfsuite
from sklearn_crfsuite import metrics

from config import Config
from data_process import process_tokens
from utils import str2float, get_predefined_split

import warnings
warnings.filterwarnings('ignore')


def featurize(word_index, word, sent, word_vector, sent_prediction, c):
    """Outputs a feature dict for a specified word

    :param word_index:
    :param word:
    :param sent:
    :param word_vector:
    :param sent_prediction:
    :param c:
    :return:
    """
    # word indexes: 0 token.text, 1 token.pos_, 2 token.tag_, 3 token.dep_, 4 str(token.vector_norm),
    #  5 str(token.cluster), 6 str(token.is_oov), 7 str(token.is_stop), 8 token.head.text, 9 token.head.pos_
    word = word.split()

    features = {'bias': 1.0,
                'word_lower': word[0].lower(),
                # 'WordShape': word[0].istitle() and word[0][1:] != word[0][1:].lower(),
                # 'word': word[0],
                # 'word[-3:]': word[0][-3:],
                # 'word[-2:]': word[0][-2:],
                # 'word.isupper()': word[0].isupper(),
                'word.istitle()': word[0].istitle(),
                # 'word.isdigit()': word[0].isdigit(),
                'pos': word[1],
                'tag': word[2],
                'dep': word[3],
                'vector_norm': float(word[4]),
                'cluster': word[5],
                # 'oov': bool(word[6]),
                'is_stop': bool(word[7]),
                'dep_head': word[8],
                'dep_head_pos': word[9]
                }

    if word_index > 0:
        prev_word = sent[word_index - 1].split()
        features.update({
            'prev_word.lower()': prev_word[0].lower(),
            # 'WordShape': word[0].istitle() and word[0][1:] != word[0][1:].lower(),
            # 'prev_word.istitle()': prev_word[0].istitle(),
            # 'prev_word.isupper()': prev_word[0].isupper(),
            # 'prev_word.isdigit()': prev_word[0].isdigit(),
            'prev_word_pos': prev_word[1],
            'prev_word_tag': prev_word[2],
            'prev_word_dep': prev_word[3],
            # 'prev_word_vector_norm': float(prev_word[4]),
            'prev_word_cluster': prev_word[5],
            # 'prev_word_oov': bool(prev_word[6]),
            'prev_word_is_stop': bool(prev_word[7])  # ,
            # 'prev_dep_head': prev_word[8]#,
            # 'prev_dep_head_pos': prev_word[9]
        })
    else:
        features['BOS'] = True

    if word_index < len(sent) - 1:
        next_word = sent[word_index + 1].split()
        features.update({
            'next_word.lower()': next_word[0].lower(),
            # 'WordShape': word[0].istitle() and word[0][1:] != word[0][1:].lower(),
            # 'next_word.istitle()': next_word[0].istitle(),
            # 'next_word.isupper()': next_word[0].isupper(),
            # 'next_word.isdigit()': next_word[0].isdigit(),
            'next_word_pos': next_word[1],
            'next_word_tag': next_word[2],
            'next_word_dep': next_word[3],
            'next_word_vector_norm': float(next_word[4]),
            'next_word_cluster': next_word[5],
            'next_word_oov': bool(next_word[6]),
            'next_word_is_stop': bool(next_word[7])  # ,
            # 'next_dep_head': next_word[8]  # ,
            # 'next_dep_head_pos': next_word[9]
        })
    else:
        features['EOS'] = True
    # features = {}
    # features['vec'] = list(word_vector)
    if c.use_word_vec:
        for i, elem in enumerate(list(word_vector)):
            features['v_' + str(i)] = float(elem)

    features['sent_prediction'] = 1.0 - float(sent_prediction)
    return features


def get_data(c):
    """Retrieves data and labels for each section of the data (i.e. train, dev, test)

    :param c: Config object containing file locations and runtime configuration settings
    :return: data, labels -- for each of train, dev, test -- and the set of all labels seen in all sets
    """
    train, train_labels, train_label_set = _get_data(c.training_data_folder, c.training_feat, c.training_vecs,
                                                     c.training_predict)
    if c.verbosity:
        print('processed train')
    dev, dev_labels, dev_label_set = _get_data(c.dev_data_folder, c.dev_feat, c.dev_vecs, c.dev_predict)
    if c.verbosity:
        print('processed dev')
    test, test_labels, test_label_set = _get_data(c.test_data_folder, c.test_feat, c.test_vecs, c.test_predict)
    if c.verbosity:
        print('processed test')
    label_set = set()
    label_set.update(*[train_label_set, dev_label_set, test_label_set])
    return train, train_labels, dev, dev_labels, test, test_labels, label_set


def _get_data(folder, feat, vecs, predict):
    """Gets data, labels for a part of the data set

    :param folder: location of data (e.g. training data folder)
    :param feat: location of compressed feature file
    :param vecs: location of compressed vec file
    :param predict: location oof compressed file containing predictions from a sentence level classier
    :return: data, labels, the set of individual labels seen in this part of the data set
    """
    _, labels, _, _ = process_tokens(folder)
    label_set = set()
    label_set.update(*list(chain(labels)))
    with open(feat, 'rb') as training:
        tokens = msgpack.unpack(training)
    with open(vecs, 'rb') as vec_training:
        vecs = msgpack.unpack(vec_training)
        vecs = [str2float(vec) for vec in vecs]
    with open(predict, 'rb') as sent_predict:
        sentence_predictions = msgpack.unpack(sent_predict)
    data = [sent2features(s, vecs[i], sentence_predictions[i]) for i, s in enumerate(tokens)]
    return data, labels, label_set


def print_relaxed_scores(dev_labels, pred_dev, test_labels, pred_test, content_labels, c):
    """Prints the relaxed scores, i.e. with out B-'s and I-'s.

    :param dev_labels: true labels for the dev set
    :param pred_dev: predicted labels for the dev set
    :param test_labels: true labels for the test set
    :param pred_test: predicted labels for the test set
    :param content_labels: all the labels seen in the data set (minus 'O')
    :param c: Config object runtime configuration settings
    """
    relaxed_pred_dev, relaxed_dev, relaxed_pred_test, relaxed_test \
        = map(relax, [pred_dev, dev_labels, pred_test, test_labels])
    relaxed_content_labels = sorted(list(set([re.sub('[BI]-', '', label) for label in content_labels])))
    _print_scores(relaxed_content_labels, relaxed_dev, relaxed_pred_dev, 'relaxed dev', c)
    _print_scores(relaxed_content_labels, relaxed_test, relaxed_pred_test, 'relaxed test', c)


def print_scores(dev_labels, pred_dev, test_labels, pred_test, label_set, c):
    """Prints scores for the dev and test sets.

    :param dev_labels: true labels for the dev set
    :param pred_dev: predicted labels for the dev set
    :param test_labels: true labels for the test set
    :param pred_test: predicted labels for the test set
    :param label_set: set of labels seen in the whole data set
    :param c: Config object runtime configuration settings
    """
    content_labels = list(label_set)
    content_labels.remove('O')
    content_labels = sorted(
        content_labels,
        key=lambda name: (name[1:], name[0])
    )
    print_strict_scores(dev_labels, pred_dev, test_labels, pred_test, content_labels, c)
    print_relaxed_scores(dev_labels, pred_dev, test_labels, pred_test, content_labels, c)


def print_strict_scores(dev_labels, pred_dev, test_labels, pred_test, content_labels, c):
    """Prints the (strict) scores for dev and test sets, i.e. keeping BIO format

    :param dev_labels: true labels for the dev set
    :param pred_dev: predicted labels for the dev set
    :param test_labels: true labels for the test set
    :param pred_test: predicted labels for the test set
    :param content_labels: all the labels seen in the data set (minus 'O')
    :param c: Config object runtime configuration settings
    """
    _print_scores(content_labels, dev_labels, pred_dev, 'dev', c)
    _print_scores(content_labels, test_labels, pred_test, 'test', c)


def _print_scores(content_labels, true_labels, pred_labels, data_section, c):
    """

    :param content_labels: all the labels seen in the data set (minus 'O')
    :param true_labels: the real labels for a section of the data
    :param pred_labels: the labels predicted by a classifier for a section of the data
    :param data_section: str naming the section of the data being scored, e.g. 'dev', 'relaxed dev'
    :param c: Config object runtime configuration settings
    """
    results = metrics.flat_f1_score(true_labels, pred_labels,
                                    average='weighted', labels=content_labels)
    print('{} results:'.format(data_section), results)
    if c.verbosity:
        print(metrics.flat_classification_report(
            true_labels, pred_labels, labels=content_labels, digits=3
        ))


def relax(labels):
    """ Realaaax and take the B- and I- prefixes off of all labels in a portion of the data set

    :param labels: list of labels for a part of the data set
    :return: a list of labels sans B- and I-
    """
    return [[re.sub('[BI]-', '', label) for label in sentence] for sentence in labels]


def sent2features(sent, sent_vecs, sent_prediction, c):
    """Gets a list features for each word in the sentence

    :param sent: list of 'words' (i.e. the features stored during pre-processing)
    :param sent_vecs: list of word vectors
    :param sent_prediction: probability that the sentence is 'relevant', according to a sentence level classifier
    :return: a list of feature dictionaries
    """
    return [featurize(i, word, sent, sent_vecs[i], sent_prediction, c) for i, word in enumerate(sent)]


def train_crf(train, train_labels, dev, dev_labels, test, c):
    """

    :param train: training data -- list of list of feature dictionaries
    :param train_labels: labels for the training set
    :param dev: dev data -- list of list of feature dictionaries
    :param dev_labels: labels for the dev set
    :param test: test data -- list of list of feature dictionaries
    :param c: Config object containing parameters for model
    :return: trained crf model, label predictions for dev and test sets
    """
    if c.crf_parameter_search:
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            all_possible_transitions=True
        )
        params_space = {
            'c1': scipy.stats.expon(scale=0.5),
            'c2': scipy.stats.expon(scale=0.05),
            # 'linesearch': ('MoreThuente', 'Backtracking', 'StrongBacktracking'),
            # 'max_iterations': (75, 100, 125, 150, 175)
        }
        f1_scorer = make_scorer(metrics.flat_f1_score, average='weighted')  # , labels=labels)
        predefined_split = get_predefined_split(train, dev)
        crf = RandomizedSearchCV(crf, params_space,
                                 cv=predefined_split,
                                 verbose=1,
                                 n_jobs=-1,
                                 n_iter=3,
                                 scoring=f1_scorer)
        t = time.time()
        crf.fit(train + dev, train_labels + dev_labels)
        print('fit data in ' + str(time.time() - t) + "s")
        print('best params:', crf.best_params_)
        print('best CV score:', crf.best_score_)
        print('model size: {:0.2f}M'.format(crf.best_estimator_.size_ / 1000000))

    else:
        # optimized for use without word vecs
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.521871212871677,  # c1=0.7508330047195315
            c2=0.0003952592781021964,  # c2=0.03347864314032029
            # linesearch='StrongBacktracking',
            max_iterations=100,
            all_possible_transitions=True,
            # all_possible_states=True
        )
        t = time.time()
        crf.fit(train, train_labels)
        print('fit data in ' + str(time.time() - t) + "s")
    dev_pred = crf.predict(dev)
    test_pred = crf.predict(test)

    return crf, dev_pred, test_pred


def main():
    c = Config()

    train, train_labels, dev, dev_labels, test, test_labels, label_set = get_data(c)

    crf, pred_dev, pred_test = train_crf(train, train_labels, dev, dev_labels, test, c)

    print_scores(dev_labels, pred_dev, test_labels, pred_test, label_set, c)


if __name__ == '__main__':
    main()
