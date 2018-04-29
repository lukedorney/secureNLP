import os
import time
import re
from itertools import chain
import msgpack
from sklearn.metrics import make_scorer
from sklearn_crfsuite import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectKBest
import sklearn_crfsuite
from data_process import process_tokens
import scipy
import numpy as np


def main():
    folder_name = os.path.join('training_material', 'data')
    token_folder_name = os.path.join(folder_name, 'tokenized')
    _, tokens_by_sent, labels = process_tokens(token_folder_name)
    label_set = set()
    label_set.update(*list(chain(labels)))
    print(str(label_set))

    with open('train_output.msgpack', 'rb') as training:
        tokens = msgpack.unpack(training)
    with open('vec_train_output.msgpack', 'rb') as vec_training:
        vecs = msgpack.unpack(vec_training)

    #vecs = np.concatenate([np.array([np.array(list(map(float, vec.split()))) for vec in sent]) for sent in vecs], axis=0)
    vecs = [np.array([np.array(list(map(float, vec.split()))) for vec in sent]) for sent in vecs]
    #vec_reducer = SelectKBest().fit(vecs, labels)
    #vecs = vec_reducer.transform(vecs)
    #print(str(vec_reducer.get_support(indices=True)))

    with open('sent_predict.msgpack', 'rb') as sent_predict:
        sentence_predictions = msgpack.unpack(sent_predict)

    only_gold = False
    if only_gold:
        gold_nums, gold_tokens, gold_labels = get_gold(token_folder_name, tokens, labels)
        x_train = [sent2features(s, vecs[gold_nums[i]], sentence_predictions[i]) for i, s in enumerate(gold_tokens)]
        labels = gold_labels
    else:
        x_train = [sent2features(s, vecs[i], sentence_predictions[i]) for i, s in enumerate(tokens)]
    print('processed train')

    folder_name = os.path.join('dev', 'dev_source')
    token_folder_name = os.path.join(folder_name, 'tokenized')
    _, _, dev_labels = process_tokens(token_folder_name)
    label_set = set()
    label_set.update(*list(chain(dev_labels)))
    print(str(label_set))

    '''with open('dev_output.msgpack', 'rb') as dev:
        dev_tokens = msgpack.unpack(dev)
    with open('vec_dev_output.msgpack', 'rb') as vec_dev:
        dev_vecs = msgpack.unpack(vec_dev)

    dev_vecs = [[vec.split() for vec in sent] for sent in dev_vecs]
    #dev_vecs = vec_reducer.transform(dev_vecs)

    with open('dev_sent_predict.msgpack', 'rb') as dev_sent_predict:
        dev_sentence_predictions = msgpack.unpack(dev_sent_predict)

    dev_x_train = (sent2features(s, dev_vecs[i], dev_sentence_predictions[i]) for i, s in enumerate(dev_tokens))
    print('processed dev')'''

    folder_name = os.path.join('gold', 'gold_source')
    token_folder_name = os.path.join(os.path.join(folder_name, 'test_1'), 'tokenized')
    _, _, test_labels = process_tokens(token_folder_name)
    label_set = set()
    label_set.update(*list(chain(test_labels)))
    print(str(label_set))

    with open('test_1_output.msgpack', 'rb') as dev:
        test_tokens = msgpack.unpack(dev)
    with open('vec_test_1_vec_train_output.msgpack', 'rb') as vec_dev:
        test_vecs = msgpack.unpack(vec_dev)

    test_vecs = [[vec.split() for vec in sent] for sent in test_vecs]
    #dev_vecs = vec_reducer.transform(dev_vecs)

    with open('test_sent_predict.msgpack', 'rb') as dev_sent_predict:
        test_sentence_predictions = msgpack.unpack(dev_sent_predict)

    test_x_train = (sent2features(s, test_vecs[i], test_sentence_predictions[i]) for i, s in enumerate(test_tokens))
    print('processed test')

    parameter_search = False
    if parameter_search:
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            all_possible_transitions=True
        )
        params_space = {
            'c1': scipy.stats.expon(scale=0.5),
            'c2': scipy.stats.expon(scale=0.05),
            'linesearch': ('MoreThuente', 'Backtracking', 'StrongBacktracking'),
            'max_iterations': (75, 100, 125, 150, 175)
        }
        f1_scorer = make_scorer(metrics.flat_f1_score, average='weighted') #, labels=labels)
        rs = RandomizedSearchCV(crf, params_space,
                                cv=3,
                                verbose=1,
                                n_jobs=-1,
                                n_iter=55,
                                scoring=f1_scorer)
        t = time.time()
        rs.fit(x_train, labels)
        print('fit data in ' + str(time.time()-t) + "s")
        print('best params:', rs.best_params_)
        print('best CV score:', rs.best_score_)
        print('model size: {:0.2f}M'.format(rs.best_estimator_.size_ / 1000000))
        all_labels = list(rs.classes_)
        content_labels = list(rs.classes_)
        content_labels.remove('O')
        y_pred = rs.predict(test_x_train)
    else:
        # optimized for use without word vecs
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.521871212871677, #c1=0.7508330047195315
            c2=0.0003952592781021964, #c2=0.03347864314032029
            # linesearch='StrongBacktracking',
            max_iterations=100,
            all_possible_transitions=True,
            # all_possible_states=True
        )
        t = time.time()
        crf.fit(x_train, labels)
        print('fit data in ' + str(time.time() - t) + "s")
        all_labels = list(crf.classes_)
        content_labels = list(crf.classes_)
        content_labels.remove('O')
        y_pred = crf.predict(test_x_train)

    results = metrics.flat_f1_score(test_labels, y_pred,
                                    average='weighted', labels=all_labels)
    print('test results with Os -- ' + str(results))
    results = metrics.flat_f1_score(test_labels, y_pred,
                                    average='weighted', labels=content_labels)
    print('test results without Os -- ' + str(results))

    sorted_labels = sorted(
        content_labels,
        key=lambda name: (name[1:], name[0])
    )
    print(metrics.flat_classification_report(
        test_labels, y_pred, labels=sorted_labels, digits=3
    ))

    #relaxedscore
    relaxed_predicted = [[re.sub('[BI]-','',word) for word in sentence] for sentence in y_pred]
    relaxed_test = [[re.sub('[BI]-','',word) for word in sentence] for sentence in test_labels]
    relaxed_all_labels = list(set([re.sub('[BI]-', '', label) for label in label_set]))
    relaxed_content_labels = list(set([re.sub('[BI]-', '', label) for label in label_set]))
    relaxed_content_labels.remove('O')
    results = metrics.flat_f1_score(relaxed_test, relaxed_predicted,
                                    average='weighted', labels=relaxed_all_labels)
    print('relaxed test results with Os -- ' + str(results))
    results = metrics.flat_f1_score(relaxed_test, relaxed_predicted,
                                    average='weighted', labels=relaxed_content_labels)
    print('relaxed test results without Os -- ' + str(results))

    sorted_labels = sorted(
        relaxed_content_labels,
        key=lambda name: (name[1:], name[0])
    )
    print(metrics.flat_classification_report(
        relaxed_test, relaxed_predicted, labels=sorted_labels, digits=3
    ))


def featurize(word_index, word, sent, word_vector, sent_prediction, use_word_vec=False):
    # 0 token.text, 1 token.pos_, 2 token.tag_, 3 token.dep_, 4 str(token.vector_norm), 5 str(token.cluster),
    #  6 str(token.is_oov), 7 str(token.is_stop), 8 token.head.text, 9 token.head.pos_
    word = word.split()

    features = {'bias': 1.0,
                'word_lower': word[0].lower(),
                #'WordShape': word[0].istitle() and word[0][1:] != word[0][1:].lower(),
                #'word': word[0],
                #'word[-3:]': word[0][-3:],
                #'word[-2:]': word[0][-2:],
                #'word.isupper()': word[0].isupper(),
                'word.istitle()': word[0].istitle(),
                #'word.isdigit()': word[0].isdigit(),
                'pos': word[1],
                'tag': word[2],
                'dep': word[3],
                'vector_norm': float(word[4]),
                'cluster': word[5],
                #'oov': bool(word[6]),
                'is_stop': bool(word[7]),
                'dep_head': word[8],
                'dep_head_pos': word[9]
    }

    if word_index > 0:
        prev_word = sent[word_index - 1].split()
        features.update({
            'prev_word.lower()': prev_word[0].lower(),
            #'WordShape': word[0].istitle() and word[0][1:] != word[0][1:].lower(),
            #'prev_word.istitle()': prev_word[0].istitle(),
            #'prev_word.isupper()': prev_word[0].isupper(),
            #'prev_word.isdigit()': prev_word[0].isdigit(),
            'prev_word_pos': prev_word[1],
            'prev_word_tag': prev_word[2],
            'prev_word_dep': prev_word[3],
            #'prev_word_vector_norm': float(prev_word[4]),
            'prev_word_cluster': prev_word[5],
            #'prev_word_oov': bool(prev_word[6]),
            'prev_word_is_stop': bool(prev_word[7])#,
            #'prev_dep_head': prev_word[8]#,
            #'prev_dep_head_pos': prev_word[9]
        })
    else:
        features['BOS'] = True

    if word_index < len(sent) - 1:
        next_word = sent[word_index + 1].split()
        features.update({
            'next_word.lower()': next_word[0].lower(),
            #'WordShape': word[0].istitle() and word[0][1:] != word[0][1:].lower(),
            #'next_word.istitle()': next_word[0].istitle(),
            #'next_word.isupper()': next_word[0].isupper(),
            #'next_word.isdigit()': next_word[0].isdigit(),
            'next_word_pos': next_word[1],
            'next_word_tag': next_word[2],
            'next_word_dep': next_word[3],
            'next_word_vector_norm': float(next_word[4]),
            'next_word_cluster': next_word[5],
            'next_word_oov': bool(next_word[6]),
            'next_word_is_stop': bool(next_word[7])#,
            #'next_dep_head': next_word[8]  # ,
            #'next_dep_head_pos': next_word[9]
        })
    else:
        features['EOS'] = True
    # features = {}
    # features['vec'] = list(word_vector)
    if use_word_vec:
        for i, elem in enumerate(list(word_vector)):
            features['v_'+str(i)] = float(elem)

    features['sent_prediction'] = 1.0 - float(sent_prediction)
    return features


def sent2features(sent, sent_vecs, sent_prediction):
    return [featurize(i, word, sent, sent_vecs[i], sent_prediction) for i, word in enumerate(sent)]


def get_gold(token_folder_name, tokens, labels):
    _, _, gold_sent_labels = process_tokens(token_folder_name, sent_classification=True)
    gold_nums = list()
    gold_tokens = list()
    gold_labels = list()
    for i, sent in enumerate(tokens):
        if gold_sent_labels[i] == 1:
            gold_nums.append(i)
            gold_tokens.append(sent)
            gold_labels.append(labels[i])
    return gold_nums, gold_tokens, gold_labels

if __name__ == '__main__':
    main()
