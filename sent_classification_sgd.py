import msgpack
import numpy as np
# from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer  # , HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler  # , Normalizer
from sklearn.svm import SVC

from config import Config
from data_process import process_tokens
from utils import str2float, get_predefined_split


def combined_predict(text_clf, vec_clf, train, dev, test, c):
    """ Makes predictions based on a weight combination of two classifiers a specified segment of the data set.

    :param text_clf: classifier trained on text data
    :param vec_clf: classifier trained on sentence level vectors
    :param test:
    :param dev:
    :param train:
    :param c: Config object with file locations
    :return:
    """
    train_combined = _combined_predict(text_clf, vec_clf, train)
    dev_combined = _combined_predict(text_clf, vec_clf, dev)
    test_combined = _combined_predict(text_clf, vec_clf, test)
    if c.find_weights:
        dev_avg, dev_predict, optimal_weight = find_optimal_weight(dev_combined, dev['labels'], c)
        train_avg = np.average(train_combined, axis=0, weights=[optimal_weight, c.range[-1] - optimal_weight])
        test_avg = np.average(test_combined, axis=0, weights=[optimal_weight, c.range[-1] - optimal_weight])
    else:
        train_avg = np.average(train_combined, axis=0, weights=c.sent_clf_weights)
        dev_avg = np.average(dev_combined, axis=0, weights=c.sent_clf_weights)
        test_avg = np.average(test_combined, axis=0, weights=c.sent_clf_weights)
        dev_predict = np.argmax(dev_avg, axis=1) if c.verbosity else None

    if c.verbosity:
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


def find_optimal_weight(combined_probs, labels, c):
    """ Finds optimal weights for combining classifiers

    :param combined_probs: probabilties of data based on text and vector classifiers
    :param labels: labels for a portion of the data set
    :param c: Config object that contains parameter settings
    :return:
    """
    weights = list()
    for w in np.linspace(*c.range, c.num_steps):
        avg = np.average(combined_probs, axis=0, weights=[w, c.range[-1] - w])
        predict = np.argmax(avg, axis=1)
        weights.append((avg, predict, w, metrics.f1_score(labels, predict, pos_label=1)))
    combine_avg, combine_predict, optimal_weight, score = max(weights, key=lambda x: x[-1])
    if c.verbosity:
        print('Optimal weights (dev-based):\nText clf:', optimal_weight, '\nVector clf:', c.range[-1] - optimal_weight)
    return combine_avg, combine_predict, optimal_weight


def get_data(c):
    """ Retrieves data sets.

    :param c: Config object containing folder and file locations
    :return: Three data sets, each made of text data, sentence vectors, and labels
    """
    train, dev, test = dict(), dict(), dict()
    _, _, train['text'], train['labels'] = process_tokens(c.training_data_folder)
    _, _, dev['text'], dev['labels'] = process_tokens(c.dev_data_folder)
    _, _, test['text'], test['labels'] = process_tokens(c.test_data_folder)
    train['vecs'], dev['vecs'], test['vecs'] = get_vecs(c)
    return train, dev, test


def get_vecs(c):
    """ Loads sentence vectors from file

    :param c: Config object containing file locations
    :return:
    """
    vecs = list()
    for vec_file in [c.training_sent_vecs, c.dev_sent_vecs, c.test_sent_vecs]:
        with open(vec_file, 'rb') as vec:
            vec = msgpack.unpack(vec)
            vecs.append(str2float(vec))
    return vecs


def print_score(true_labels, pred_labels):
    """ Helper function to print F1 score and precision/recall/f1 chart

    :param true_labels: labels provided in the data set
    :param pred_labels: labels predicted by a classifier
    """
    print(metrics.f1_score(true_labels, pred_labels, pos_label=1))
    print(metrics.classification_report(true_labels, pred_labels))


def train_text_clf(train, dev, c):
    """ Trains a classifier based on text data

    :param train: training set
    :param dev: dev set
    :param c: Config object containing parameters
    :return: classifier trained on text data
    """
    if not c.sent_text_parameter_search:
        p = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2), min_df=6, max_df=.75)),
                      ('tfidf', TfidfTransformer(sublinear_tf=True)),
                      ('clf', SGDClassifier(loss='log', penalty='l2', alpha=1e-04, max_iter=1000))
                      ])
        # p = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2))),
        #               ('tfidf', TfidfTransformer(sublinear_tf=True)),
        #               ('scaler', StandardScaler(copy=False, with_mean=False)),
        #               # ('svd', TruncatedSVD(n_components=1000)),
        #               # ('normalizer', Normalizer(copy=False)),
        #               ('clf', SVC(C=10000, gamma=1e-09, class_weight={1: 15}, cache_size=1000))
        #               ])
        # p = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1), min_df=3, max_df=.95)),
        #               ('svd', LatentDirichletAllocation(n_components=30, n_jobs=-1, learning_method='batch', learning_offset=70., max_iter=15)),
        #               # ('normalizer', Normalizer(copy=False)),
        #               ('clf', SGDClassifier(loss='log', penalty='l2', alpha=1e-04, max_iter=1000))
        #               ])
        p.fit(train['text'], train['labels'])
        return p
    else:
        predefined_split = get_predefined_split(train['text'], dev['text'])
        p = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer(sublinear_tf=True)),
                      ('scaler', StandardScaler(copy=False, with_mean=False)),
                      # ('svd', TruncatedSVD(n_components=1000)),
                      # ('normalizer', Normalizer(copy=False)),
                      ('clf', SVC(cache_size=1000))
                      ])
        c_range = np.logspace(-2, 10, 13)
        gamma_range = np.logspace(-9, 3, 13)
        parameters = {'vect__ngram_range': [(1, 1), (1, 2)], #, (2, 1), (2, 2)],
                      # 'tfidf__sublinear_tf': (True, False),
                      'clf__C': c_range,
                      'clf__gamma': gamma_range,
                      'clf__class_weight': ({1: 1}, {1: 5}),
                      # 'clf__alpha': (.001, .0001, .0005, .005, .00001, .00005),
                      # 'clf__loss': ('hinge', 'log'),
                      # 'clf__penalty': ('elasticnet', 'l2', 'l1'),
                      # 'clf__fit_intercept': (True, False),
                      }
        gs_clf = GridSearchCV(p, parameters, cv=predefined_split, scoring='f1', n_jobs=-1)
        gs_clf = gs_clf.fit(train['text'] + dev['text'], train['labels'] + dev['labels'])
        print(gs_clf.best_score_)
        for param_name in sorted(parameters.keys()):
            print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
        return gs_clf


def train_vec_clf(train, dev, c):
    """ Trains a classifier based on sentence vectors

    :param train: training set
    :param dev: dev set
    :param c: Config object containing parameters
    :return: classifier trained on sentence vectors
    """
    if not c.sent_vec_parameter_search:
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
            # 'clf__warm_start': (True, False)
            'clf__l1_ratio': (.10, .15, .20, .25, .30, .45, .5, .55, .60, .65, .70, .75)
        }
        gs_clf = GridSearchCV(p, parameters, scoring='f1', cv=predefined_split, n_jobs=-1)
        gs_clf = gs_clf.fit(train['vecs'] + dev['vecs'], train['labels'] + dev['labels'])
        print(gs_clf.best_score_)
        for param_name in sorted(parameters.keys()):
            print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
        return gs_clf


def write_sent_prob(train, dev, test, c):
    """ Writes a classifiers sentence level prediction probabilities to file to use as feature in NER.

    :param train: predictions for training set if precomputed else training set
    :param dev: predictions for dev set if precomputed else dev set
    :param test: predictions for test set if precomputed else test set
    :param c: Config object with file locations
    """
    for predictions, prediction_file in zip([train, dev, test], [c.training_predict, c.dev_predict, c.test_predict]):
        with open(prediction_file, 'wb') as sent_predict:
            msgpack.pack(predictions[:, 0], sent_predict)


def main():
    c = Config()
    train, dev, test = get_data(c)
    text_clf = train_text_clf(train, dev, c)
    vec_clf = train_vec_clf(train, dev, c)

    print('---Text classifier---')
    dev_test_predict(text_clf, dev, test)
    print('---Vector classifier---')
    dev_test_predict(vec_clf, dev, test, text=False)

    if c.sent_combine_models:
        train_prob, dev_prob, test_prob = combined_predict(text_clf, vec_clf, train, dev, test, c)
    else:
        train_prob, dev_prob, test_prob = map(text_clf.predict_proba, [train['text'], dev['text'], test['text']])
    if c.write_probs:
        write_sent_prob(train_prob, dev_prob, test_prob, c)


if __name__ == '__main__':
    main()
