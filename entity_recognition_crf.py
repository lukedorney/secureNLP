"""Creates classifiers for addressing task 2 of named-entity recognition.

Uses sklearn_crfsuite to create a model capable of structured prediction.
Relies on preprocessed data for features and vectors and on task 1 classification probabilities on the sent level.

Scoring comes in two varieties: 'strict' and 'relaxed' F1 scoring, as put forth in the tasks ACL paper.
Strict scoring is based on the full tag in the data (i.e. in BIO format) -- e.g. B-Entity, I-Modifier.
Relaxed scoring is based on the part of the tag following the BIO label -- e.g. Entity, Modifier.
"""
from itertools import chain
import re
import time

from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import sklearn_crfsuite
from sklearn_crfsuite import metrics

from config import Config
from utils import str2float, get_predefined_split, read_msgpack, relax

import warnings
warnings.filterwarnings('ignore')


CONFIG = Config()


def featurize(word_index, word, sent, word_vector, sent_prediction):
    """Outputs a feature dict for a specified word

    :param word_index: index of the word in the sentence
    :param word: features for a word
    :param sent: sentence (list of word features) word is located in
    :param word_vector: the vector for word
    :param sent_prediction: probability provided by the sentence level classifier
    :return: a dict of features for a word
    """
    # word indexes: 0 token.text, 1 token.pos_, 2 token.tag_, 3 token.dep_, 4 str(token.vector_norm),
    # 5 str(token.cluster), 6 str(token.is_oov), 7 str(token.is_stop), 8 token.head.text, 9 token.head.pos_,
    # 10 token.lemma, 11 str(token.like_email), 12 str(token.like_url), 13 neglog prob, 14 ent-iob, 15 ent-type,
    # 16 shape
    word = word.split()

    features = {'bias': 1.0,
                'word_lower': word[0].lower(),
                'word.istitle()': word[0].istitle(),
                'pos': word[1],
                'tag': word[2],
                'dep': word[3],
                'vector_norm': float(word[4]),
                'cluster': word[5],
                'is_stop': bool(word[7]),
                'dep_head': word[8],
                'dep_head_pos': word[9],
                }

    if word_index > 0:
        prev_word = sent[word_index - 1].split()
        features.update({
            'prev_word.lower()': prev_word[0].lower(),
            'prev_word_pos': prev_word[1],
            'prev_word_tag': prev_word[2],
            'prev_word_dep': prev_word[3],
            'prev_word_cluster': prev_word[5],
            'prev_word_is_stop': bool(prev_word[7]),
        })
    else:
        features['BOS'] = True

    if word_index < len(sent) - 1:
        next_word = sent[word_index + 1].split()
        features.update({
            'next_word.lower()': next_word[0].lower(),
            'next_word_pos': next_word[1],
            'next_word_tag': next_word[2],
            'next_word_dep': next_word[3],
            'next_word_vector_norm': float(next_word[4]),
            'next_word_cluster': next_word[5],
            'next_word_oov': bool(next_word[6]),
            'next_word_is_stop': bool(next_word[7]),
            # 'word_dep_comb': next_word[0].lower() + '-' + next_word[3]
        })
    else:
        features['EOS'] = True

    if CONFIG.use_word_vec:
        for i, elem in enumerate(list(word_vector)):
            features['v_' + str(i)] = float(elem)
    if CONFIG.use_sentence_prediction:
        features['sent_prediction'] = 1.0 - float(sent_prediction)
    return features


def get_data():
    """Retrieves data and labels for each section of the data (i.e. train, dev, test)

    :return: data, labels -- for each of train, dev, test -- and the set of all labels seen in all sets
    """
    train, train_labels, train_label_set = _get_data(CONFIG.train_labels_by_sent, CONFIG.train_feat, CONFIG.train_vec,
                                                     CONFIG.train_predict)
    if CONFIG.verbose:
        print('processed train')
    dev, dev_labels, dev_label_set = _get_data(CONFIG.dev_labels_by_sent, CONFIG.dev_feat, CONFIG.dev_vec,
                                               CONFIG.dev_predict)
    if CONFIG.verbose:
        print('processed dev')
    test, test_labels, test_label_set = _get_data(CONFIG.test_labels_by_sent, CONFIG.test_feat, CONFIG.test_vec,
                                                  CONFIG.test_predict)
    if CONFIG.verbose:
        print('processed test')
    label_set = set()
    label_set.update(*[train_label_set, dev_label_set, test_label_set])
    return train, train_labels, dev, dev_labels, test, test_labels, label_set


def _get_data(labels, feat, vec, predict):
    """Gets data, labels for a part of the data set

    :param labels: location of compressed labels
    :param feat: location of compressed feature file
    :param vec: location of compressed vec file
    :param predict: location oof compressed file containing predictions from a sentence level classier
    :return: data, labels, the set of individual labels seen in this part of the data set
    """
    labels = read_msgpack(labels)
    label_set = set()
    label_set.update(*list(chain(labels)))
    tokens = read_msgpack(feat)
    vec = read_msgpack(vec)
    vec = [str2float(vec) for vec in vec]
    sentence_predictions = read_msgpack(predict)
    data = [sent2features(s, vec[i], sentence_predictions[i]) for i, s in enumerate(tokens)]
    return data, labels, label_set


def predict_test_comb(crf):
    """Makes predictions for combined .in test file.

    :param crf: trained crf classifier
    """
    tokens = read_msgpack(CONFIG.task_2_comb_tokens)
    features = read_msgpack(CONFIG.task_2_comb_feat)
    vec = read_msgpack(CONFIG.task_2_comb_vec)
    vec = [str2float(v) for v in vec]
    sentence_predictions = read_msgpack(CONFIG.test_comb_predict)
    data = [sent2features(s, vec[i], sentence_predictions[i]) for i, s in enumerate(features)]
    pred_test_comb = crf.predict(data)
    write_test_results(tokens, pred_test_comb, CONFIG.task2_out)


def print_relaxed_scores(dev_labels, pred_dev, test_labels, pred_test, content_labels):
    """Prints the relaxed scores, i.e. with out B-'s and I-'s.

    :param dev_labels: true labels for the dev set
    :param pred_dev: predicted labels for the dev set
    :param test_labels: true labels for the test set
    :param pred_test: predicted labels for the test set
    :param content_labels: all the labels seen in the data set (minus 'O')
    """
    relaxed_pred_dev, relaxed_dev, relaxed_pred_test, relaxed_test \
        = map(relax, [pred_dev, dev_labels, pred_test, test_labels])
    relaxed_content_labels = sorted(list(set([re.sub('[BI]-', '', label) for label in content_labels])))
    _print_scores(relaxed_content_labels, relaxed_dev, relaxed_pred_dev, 'relaxed dev')
    _print_scores(relaxed_content_labels, relaxed_test, relaxed_pred_test, 'relaxed test')


def print_scores(dev_labels, pred_dev, test_labels, pred_test, label_set):
    """Prints scores for the dev and test sets.

    :param dev_labels: true labels for the dev set
    :param pred_dev: predicted labels for the dev set
    :param test_labels: true labels for the test set
    :param pred_test: predicted labels for the test set
    :param label_set: set of labels seen in the whole data set
    """
    content_labels = list(label_set)
    content_labels.remove('O')
    content_labels = sorted(
        content_labels,
        key=lambda name: (name[1:], name[0])
    )
    print_strict_scores(dev_labels, pred_dev, test_labels, pred_test, content_labels)
    print_relaxed_scores(dev_labels, pred_dev, test_labels, pred_test, content_labels)


def print_strict_scores(dev_labels, pred_dev, test_labels, pred_test, content_labels):
    """Prints the (strict) scores for dev and test sets, i.e. keeping BIO format

    :param dev_labels: true labels for the dev set
    :param pred_dev: predicted labels for the dev set
    :param test_labels: true labels for the test set
    :param pred_test: predicted labels for the test set
    :param content_labels: all the labels seen in the data set (minus 'O')
    """
    _print_scores(content_labels, dev_labels, pred_dev, 'dev')
    _print_scores(content_labels, test_labels, pred_test, 'test')


def _print_scores(content_labels, true_labels, pred_labels, data_section):
    """

    :param content_labels: all the labels seen in the data set (minus 'O')
    :param true_labels: the real labels for a section of the data
    :param pred_labels: the labels predicted by a classifier for a section of the data
    :param data_section: str naming the section of the data being scored, e.g. 'dev', 'relaxed dev'
    """
    results = metrics.flat_f1_score(true_labels, pred_labels,
                                    average='weighted', labels=content_labels)
    print('{} results:'.format(data_section), results)
    if CONFIG.verbose:
        print(metrics.flat_classification_report(
            true_labels, pred_labels, labels=content_labels, digits=3
        ))


def sent2features(sent, sent_vecs, sent_prediction):
    """Gets a list features for each word in the sentence

    :param sent: list of 'words' (i.e. the features stored during pre-processing)
    :param sent_vecs: list of word vectors
    :param sent_prediction: probability that the sentence is 'relevant', according to a sentence level classifier
    :return: a list of feature dictionaries
    """
    return [featurize(i, word, sent, sent_vecs[i], sent_prediction) for i, word in enumerate(sent)]


def train_crf(train, train_labels, dev, dev_labels, test):
    """Trains a crf classifier using sklearn crf_suite.

    :param train: training data -- list of list of feature dictionaries
    :param train_labels: labels for the training set
    :param dev: dev data -- list of list of feature dictionaries
    :param dev_labels: labels for the dev set
    :param test: test data -- list of list of feature dictionaries
    :return: trained crf model, label predictions for dev and test sets
    """
    if CONFIG.crf_parameter_search:
        # crf = sklearn_crfsuite.CRF(
        #     algorithm='lbfgs',
        #     all_possible_transitions=True,
        #     max_iterations=100,
        #     c1=.521871212,
        #     c2=.000395259,
        # )
        # params_space = {
        #     'c1': scipy.stats.expon(scale=0.5),
        #     'c2': scipy.stats.expon(scale=0.05),
        #     # 'linesearch': ('MoreThuente', 'Backtracking', 'StrongBacktracking'),
        #     # 'max_iterations': (75, 100, 125, 150, 175)
        # }
        crf = sklearn_crfsuite.CRF(
            algorithm='pa',
            # max_iterations=150,
            # epsilon=1e-5,
            # pa_type=1,  # 1, 2
            # CONFIG=1,
            # error_sensitive=True,
            # averaging=True,
            # all_possible_transitions=True,
            all_possible_states=True
        )
        params_space = {
            'CONFIG': [.001, .01, .5, .1, 1, 10, 50, 100],
            'max_iterations': [50, 100, 150, 200],
        }
        f1_scorer = make_scorer(metrics.flat_f1_score, average='weighted')  # , labels=labels)
        if CONFIG.use_predefined_split:
            predefined_split = get_predefined_split(train, dev)
            # crf = RandomizedSearchCV(crf, params_space,
            #                          cv=predefined_split,
            #                          # iid=False,
            #                          verbose=1,
            #                          n_jobs=3,
            #                          n_iter=3,
            #                          scoring=f1_scorer)
            crf = GridSearchCV(crf, params_space, scoring=f1_scorer, n_jobs=6, cv=predefined_split, verbose=1)

        else:
            # crf = RandomizedSearchCV(crf, params_space,
            #                          cv=CONFIG.cv_size,
            #                          verbose=1,
            #                          n_jobs=3,
            #                          n_iter=50,
            #                          scoring=f1_scorer)
            crf = GridSearchCV(crf, params_space, scoring=f1_scorer, n_jobs=6, cv=CONFIG.cv_size, verbose=1)

        t = time.time()
        crf.fit(train + dev, train_labels + dev_labels)
        print('fit data in ' + str(time.time() - t) + "s")
        print('best params:', crf.best_params_)
        print('best CV score:', crf.best_score_)
        print('model size: {:0.2f}M'.format(crf.best_estimator_.size_ / 1000000))

    else:
        # optimized for use without word vecs
        # crf = sklearn_crfsuite.CRF(
        #     algorithm='lbfgs',
        #     c1=.521871212, #0.521871212871677,  # c1=0.7508330047195315
        #     c2=.000395259, #0.0003952592781021964,  # c2=0.03347864314032029
        #     max_iterations=100,
        #     all_possible_transitions=True,
        #     # all_possible_states=True
        # )
        crf = sklearn_crfsuite.CRF(
            algorithm='pa',
            max_iterations=50,
            epsilon=1e-5,
            pa_type=2,
            c=50,
            error_sensitive=True,
            averaging=True,
            all_possible_transitions=True,
            all_possible_states=True
        )
        t = time.time()
        crf.fit(train, train_labels)
        print('fit data in ' + str(time.time() - t) + "s")
    train_pred = crf.predict(train)
    dev_pred = crf.predict(dev)
    test_pred = crf.predict(test)

    return crf, train_pred, dev_pred, test_pred


def write_test_results(text, predicted, task2_out):
    """Writes to file predictions for combined .in test file.

    :param text: list of list of tokens
    :param predicted: list of list of labels
    :param task2_out: name of output file
    :return:
    """
    with open(task2_out, 'w', encoding='utf-8') as f:
        for sent, sent_labels in zip(text, predicted):
            for word, label in zip(sent, sent_labels):
                f.write("{}\t{}\n".format(word, label))
            f.write('\n')


def main():
    train, train_labels, dev, dev_labels, test, test_labels, label_set = get_data()

    crf, pred_train, pred_dev, pred_test_comb = train_crf(train, train_labels, dev, dev_labels, test)

    print_scores(dev_labels, pred_dev, test_labels, pred_test_comb, label_set)

    if CONFIG.write_test_results:
        predict_test_comb(crf)


if __name__ == '__main__':
    main()
