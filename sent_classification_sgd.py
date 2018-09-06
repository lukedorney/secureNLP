"""Creates classifiers for addressing task 1 of sentence classification.

Expects that data_process.py has already been run.
This file is capable of creating 2 different types of classifiers, based on the type of data input:
    (1) Text data, which is vectorized into counts, and usually transformed into tf-idf vectors
    (2) Sentence level vectors, produced by spacy

For (1), tried a combination of linear sgd methods (logistic regression, linear svm, etc) and non-linear SVM, along with
different decomposition techniques, although it seems the most successful approach is plain logistic regression.
For(2), tried different linear approaches, linear svm seems to work best, but considerably underperforms approach (1).

Explored averaging models, but doesn't seem to help as the sentence level vector clfs do not perform well on their own.

XGBoost models tried, can be successful but ultimately unstable (f1 between 47 and 56), but data seems too small

Writes probabilities of each data point to file for use by NER crf classifier.
"""
import numpy as np
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer  # , HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler  # , Normalizer
from sklearn.svm import SVC
from xgboost import XGBClassifier

from config import Config
from utils import str2float, get_predefined_split, read_msgpack, write_msgpack


CONFIG = Config()


def combined_predict(text_clf, vec_clf, train, dev, test):
    """ Makes predictions based on a weight combination of two classifiers a specified segment of the data set.

    :param text_clf: classifier trained on text data
    :param vec_clf: classifier trained on sentence level vectors
    :param train: training text and vector data
    :param dev: dev text and vector data
    :param test: test text and vector data
    :return: combined probabilities for test, dev, and test sets
    """
    train_combined = _combined_predict(text_clf, vec_clf, train)
    dev_combined = _combined_predict(text_clf, vec_clf, dev)
    test_combined = _combined_predict(text_clf, vec_clf, test)
    if CONFIG.find_weights:
        dev_avg, dev_predict, optimal_weight = find_optimal_weight(dev_combined, dev['labels'])
        train_avg = np.average(train_combined, axis=0, weights=[optimal_weight, CONFIG.range[-1] - optimal_weight])
        test_avg = np.average(test_combined, axis=0, weights=[optimal_weight, CONFIG.range[-1] - optimal_weight])
    else:
        train_avg = np.average(train_combined, axis=0, weights=CONFIG.sent_clf_weights)
        dev_avg = np.average(dev_combined, axis=0, weights=CONFIG.sent_clf_weights)
        test_avg = np.average(test_combined, axis=0, weights=CONFIG.sent_clf_weights)
        dev_predict = np.argmax(dev_avg, axis=1) if CONFIG.verbose else None

    if CONFIG.verbose:
        test_predict = np.argmax(test_avg, axis=1)
        print("---Combined classifier---\nDev Results:")
        print_score(dev['labels'], dev_predict)
        print("Test Results:")
        print_score(test['labels'], test_predict)
    return train_avg, dev_avg, test_avg


def _combined_predict(text_clf, vec_clf, data):
    """ Helper function for combined classifier prediction.

    :param text_clf: classifier trained on text data
    :param vec_clf: classifier trained on sentence vectors
    :param data: a particular subset of the dataset (e.g. training, dev)
    :return: numpy array containing probabilities from each classifier for each data point
    """
    return np.asarray([clf.predict_proba(data) for clf, data in zip([text_clf, vec_clf], [data['text'], data['vecs']])])


def dev_test_predict(clf, dev, test, text=True):
    """ Given a classifier, makes predictions on the dev and test sets

    :param clf: classifier for making predictions
    :param dev: development set dict
    :param test: test set dict
    :param text: whether to use text or vec data
    """
    if text:
        dev_pred = clf.predict(dev['text'])
        test_pred = clf.predict(test['text'])
    else:
        dev_pred = clf.predict(dev['vecs'])
        test_pred = clf.predict(test['vecs'])

    print('Dev results:')
    print_score(dev['labels'], dev_pred)
    print('Test results:')
    print_score(test['labels'], test_pred)


def find_optimal_weight(combined_probability, labels):
    """ Finds optimal weights for combining classifiers

    :param combined_probability: probabilities of data based on text and vector classifiers
    :param labels: labels for a portion of the data set
    :return: probabilities of best weighting, labels from best weighting, optimal weights
    """
    weights = list()
    for w in np.linspace(*CONFIG.range, CONFIG.num_steps):
        avg = np.average(combined_probability, axis=0, weights=[w, CONFIG.range[-1] - w])
        predict = np.argmax(avg, axis=1)
        weights.append((avg, predict, w, metrics.f1_score(labels, predict, pos_label=1)))
    combine_avg, combine_predict, optimal_weight, score = max(weights, key=lambda x: x[-1])
    if CONFIG.verbose:
        print('Optimal weights (dev-based):\nText clf:', optimal_weight, '\nVector clf:',
              CONFIG.range[-1] - optimal_weight)
    return combine_avg, combine_predict, optimal_weight


def get_data():
    """ Retrieves data sets.

    :return: Three data sets, each made of text data, sentence vectors, and labels
    """
    train, dev, test = dict(), dict(), dict()
    train['text'] = read_msgpack(CONFIG.train_sent_text)
    train['labels'] = read_msgpack(CONFIG.train_sent_labels)
    dev['text'] = read_msgpack(CONFIG.dev_sent_text)
    dev['labels'] = read_msgpack(CONFIG.dev_sent_labels)
    test['text'] = read_msgpack(CONFIG.test_sent_text)
    test['labels'] = read_msgpack(CONFIG.test_sent_labels)
    train['vecs'], dev['vecs'], test['vecs'] = get_vec()
    return train, dev, test


def get_vec():
    """ Loads sentence vectors from file.

    :return: list of sentence vectors for train, dev, and test
    """
    vec_list = list()
    for vec_file in [CONFIG.train_sent_vec, CONFIG.dev_sent_vec, CONFIG.test_sent_vec]:
        vec = read_msgpack(vec_file)
        vec_list.append(str2float(vec))
    return vec_list


def print_score(true_labels, pred_labels):
    """ Helper function to print F1 score and precision/recall/f1 chart

    :param true_labels: labels provided in the data set
    :param pred_labels: labels predicted by a classifier
    """
    print(metrics.f1_score(true_labels, pred_labels, pos_label=1))
    print(metrics.classification_report(true_labels, pred_labels, digits=3))


def train_text_clf(train, dev):
    """ Trains a classifier based on text data

    :param train: training set
    :param dev: dev set
    :return: classifier trained on text data
    """
    if not CONFIG.sent_text_parameter_search:

        # p = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2), min_df=2, max_df=.85)),
        #               ('tfidf', TfidfTransformer(sublinear_tf=True)),
        #               ('svd', TruncatedSVD(n_components=300)),
        #               ('clf', XGBClassifier(n_jobs=-1, learning_rate=0.1, n_estimators=600, max_depth=8,
        #                                     min_child_weight=0, gamma=0, subsample=0.8, colsample_bytree=0.9))
        #               ])
        p = Pipeline([('vect', CountVectorizer(ngram_range=(1, 3), min_df=10, max_df=.9, lowercase=False)),
                      ('tfidf', TfidfTransformer(sublinear_tf=True)),
                      ('clf', SGDClassifier(loss='log', penalty='l2', alpha=1e-04, max_iter=1900, class_weight={1: .9},
                                            average=True))
                      ])
        # p = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
        #               ('tfidf', TfidfTransformer(sublinear_tf=True)),
        #               # svm supposedly works better with scaled data
        #               ('scaler', StandardScaler(copy=False, with_mean=False)),
        #               # ('svd', TruncatedSVD(n_components=1000)),
        #               # ('normalizer', Normalizer(copy=False)),
        #               ('clf', SVC(C=10000, gamma=1e-09, class_weight={1: 15}, cache_size=1000))
        #               ])
        # p = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1), min_df=3, max_df=.95)),
        #               # lda works on counts -- no tf-idf transformation needed
        #               ('lda', LatentDirichletAllocation(n_components=30, n_jobs=-1, learning_method='batch',
        #                                                 learning_offset=70., max_iter=15)),
        #               # ('normalizer', Normalizer(copy=False)),
        #               ('clf', SGDClassifier(loss='log', penalty='l2', alpha=1e-04, max_iter=1000))
        #               ])
        p.fit(train['text'], train['labels'])
        return p
    else:
        predefined_split = get_predefined_split(train['text'], dev['text'])
        p = Pipeline([('vect', CountVectorizer(ngram_range=(1, 3), min_df=10, max_df=.9, lowercase=False)),
                      ('tfidf', TfidfTransformer(sublinear_tf=True)),
                      ('svd', TruncatedSVD()),
                      ('clf', SGDClassifier(loss='log', penalty='l2', alpha=1e-04, max_iter=1900, class_weight={1: .9},
                                            average=True))
                      ])
        parameters = {'svd__n_components': [100, 200, 300, 400, 500, 600, 700],
                      'svd__n_iter': [2, 3, 5, 7, 9, 11, 15]
                      }
        # c_range = np.logspace(-2, 10, 13)
        # gamma_range = np.logspace(-9, 3, 13)
        # parameters = {'vect__ngram_range': [(1, 1), (1, 2)], #, (2, 1), (2, 2)],
        #               # 'tfidf__sublinear_tf': (True, False),
        #               'clf__C': c_range,
        #               'clf__gamma': gamma_range,
        #               'clf__class_weight': ({1: 1}, {1: 5}),
        #               # 'clf__alpha': (.001, .0001, .0005, .005, .00001, .00005),
        #               # 'clf__loss': ('hinge', 'log'),
        #               # 'clf__penalty': ('elasticnet', 'l2', 'l1'),
        #               # 'clf__fit_intercept': (True, False),
        #               }
        # p = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2), min_df=2, max_df=.85)),
        #               ('tfidf', TfidfTransformer(sublinear_tf=True)),
        #               ('svd', TruncatedSVD(n_components=300)),
        #               ('clf', XGBClassifier(n_jobs=3, learning_rate=0.1, n_estimators=1000, max_depth=8,
        #                                     min_child_weight=0, gamma=0, subsample=0.8, colsample_bytree=0.9))
        #               ])
        # parameters = {'clf__gamma': [i/10.0 for i in range(0, 5, 1)]}
        gs_clf = GridSearchCV(p, parameters, cv=predefined_split, scoring='f1', n_jobs=-1)
        gs_clf = gs_clf.fit(train['text'] + dev['text'], train['labels'] + dev['labels'])
        print(gs_clf.cv_results_)
        print(gs_clf.best_score_)
        for param_name in sorted(parameters.keys()):
            print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
        return gs_clf


def train_vec_clf(train, dev):
    """ Trains a classifier based on sentence vectors.

    :param train: training set
    :param dev: dev set
    :return: classifier trained on sentence vectors
    """
    if not CONFIG.sent_vec_parameter_search:
        v = SGDClassifier(loss='log', penalty='elasticnet', max_iter=1000, l1_ratio=.65)
        v.fit(train['vecs'], train['labels'])
        return v
    else:
        predefined_split = get_predefined_split(train['vecs'], dev['vecs'])
        p = Pipeline([
            ('clf', SGDClassifier(penalty='elasticnet', max_iter=1000))
        ])
        parameters = {
            # 'clf__alpha': (.0001, .0005, .005, .00001, .00005),
            # 'clf__loss': ('log', 'perceptron'),
            # 'clf__penalty': ('elasticnet', 'l2', 'l1'),
            'clf__l1_ratio': (.10, .15, .20, .25, .30, .45, .5, .55, .60, .65, .70, .75)
        }
        gs_clf = GridSearchCV(p, parameters, scoring='f1', cv=predefined_split, n_jobs=-1)
        gs_clf = gs_clf.fit(train['vecs'] + dev['vecs'], train['labels'] + dev['labels'])
        print(gs_clf.best_score_)
        for param_name in sorted(parameters.keys()):
            print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
        return gs_clf


def write_sent_prob(train, dev, test, test_comb):
    """ Writes a classifiers sentence level prediction probabilities to file to use as feature in NER.

    :param train: predictions for training set if precomputed else training set
    :param dev: predictions for dev set if precomputed else dev set
    :param test: predictions for test set if precomputed else test set
    :param test_comb: predictions for test set from single .in file
    """
    for predictions, prediction_file in zip([train, dev, test, test_comb],
                                            [CONFIG.train_predict, CONFIG.dev_predict,
                                             CONFIG.test_predict, CONFIG.test_comb_predict]):
        write_msgpack(predictions[:, 0], prediction_file)


def write_test_results(predicted, out_file):
    """Writes results for predictions made on combined .in test file.

    :param predicted: labels predicted
    :param out_file: name of the file to be written to
    :return:
    """
    with open(out_file, 'w') as f:
        for label in predicted:
            f.write("%s\n" % label)


def main():
    train, dev, test = get_data()
    text_clf = train_text_clf(train, dev)
    vec_clf = train_vec_clf(train, dev)

    print('---Text classifier---')
    dev_test_predict(text_clf, dev, test)
    print('---Vector classifier---')
    dev_test_predict(vec_clf, dev, test, text=False)

    if CONFIG.sent_combine_models:
        train_prob, dev_prob, test_prob = combined_predict(text_clf, vec_clf, train, dev, test)
    else:
        train_prob, dev_prob, test_prob = map(text_clf.predict_proba, [train['text'], dev['text'], test['text']])

    test_combined = read_msgpack(CONFIG.task_1_comb)
    test_combined_prob = text_clf.predict_proba(test_combined)

    if CONFIG.write_sent_prob:
        write_sent_prob(train_prob, dev_prob, test_prob, test_combined_prob)

    if CONFIG.write_test_results:
        write_test_results(text_clf.predict(test_combined), CONFIG.task1_out)


if __name__ == '__main__':
    main()
