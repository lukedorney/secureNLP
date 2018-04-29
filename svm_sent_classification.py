from data_process import process_tokens
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, PredefinedSplit
from sklearn.ensemble import VotingClassifier
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.metrics import make_scorer
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import msgpack
import os
''''''
class InfoExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, loc=0):
        self.loc = loc

    def fit(self, X=None, y=None):
        print(X)
        X = X[self.loc]
        print(X)
        X = check_array(X)
        self.input_shape_ = X.shape
        return self

    def transform(self, X):
        check_is_fitted(self, ['input_shape_'])
        X = X[self.loc]
        # Input validation
        X = check_array(X)

        # Check that the input is of the same shape as the one passed
        # during fit.
        if X.shape != self.input_shape_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        return X
''''''



def main():
    folder_name = Path('training_material') / 'data'
    token_folder_name = Path(folder_name) / 'tokenized'
    _, tokens_by_sent, labels = process_tokens(token_folder_name, sent_classification=True)

    folder_name = Path('dev') / 'dev_source'
    token_folder_name = Path(folder_name) / 'tokenized'
    _, dev_tokens_by_sent, dev_labels = process_tokens(token_folder_name, sent_classification=True)

    folder_name = os.path.join('gold', 'gold_source')
    task_1_token_folder_name = os.path.join(os.path.join(folder_name, 'test_1'), 'tokenized')
    _, test_tokens_by_sent, test_labels = process_tokens(task_1_token_folder_name, sent_classification=True)

    with open('vec_sent_vec_train_output.msgpack', 'rb') as vec_training:
        vecs = msgpack.unpack(vec_training)
    vecs = [[float(v) for v in vec.split()] for vec in vecs]

    with open('vec_sent_vec_dev_output.msgpack', 'rb') as vec_dev:
        dev_vecs = msgpack.unpack(vec_dev)
    dev_vecs = [[float(v) for v in vec.split()] for vec in dev_vecs]

    with open('vec_test_1_sent_vec_train_output.msgpack', 'rb') as vec_dev:
        dev_vecs = msgpack.unpack(vec_dev)
    test_vecs = [[float(v) for v in vec.split()] for vec in dev_vecs]

    p = train_sent_classifier(tokens_by_sent, labels, parameter_search=False)
    #v = train_by_doc_vec(vecs, labels, dev_vecs=dev_vecs, dev_labels=dev_labels, parameter_search=False)

    y_pred = p.predict_proba(test_tokens_by_sent)
    #v_predict = v.predict_proba(dev_vecs)

    #combine = np.asarray([clf.predict_proba(X) for clf, X in zip([p,v], [dev_tokens_by_sent,dev_vecs])])
    #combine_avg = np.average(combine, axis=0,) # weights=[1.66, 0.33])
    #y_pred = np.argmax(combine_avg, axis=1)
    '''with open('sent_predict.msgpack', 'wb') as sent_predict:
        pred = p.predict_proba(tokens_by_sent)
        msgpack.pack([prob[0] for prob in pred], sent_predict)

    with open('dev_sent_predict.msgpack', 'wb') as sent_predict:
        msgpack.pack([prob[0] for prob in y_pred], sent_predict)'''
    with open('test_sent_predict.msgpack', 'wb') as sent_predict:
        msgpack.pack([prob[0] for prob in y_pred], sent_predict)
    '''
    word_pipe = Pipeline([
        ('info_extract', InfoExtractor(loc=0)),
        ('clf', p)
    ])
    vec_pipe = ([
        ('info_extract', InfoExtractor(loc=1)),
        ('clf', v)
    ])

    eclf = VotingClassifier(estimators=[('words', word_pipe), ('vecs', vec_pipe)], voting='hard')
    eclf.fit([tokens_by_sent, vecs],labels)
    y_pred = eclf.predict((dev_tokens_by_sent, dev_vecs))'''
    y_pred = p.predict(test_tokens_by_sent)
    # acc = np.mean(y_pred == dev_labels)
    # print((str(acc)))
    print(metrics.f1_score(test_labels,y_pred, pos_label=1))
    print(metrics.classification_report(test_labels, y_pred))


def train_sent_classifier(tokens_by_sent, labels, parameter_search=False):
    if not parameter_search:

        p = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1))),
                      ('tfidf', TfidfTransformer(sublinear_tf=True)),
                      ('clf', SGDClassifier(warm_start=False, loss='log', penalty='l2', alpha=1e-04, max_iter=10))
                      # ('clf', BernoulliNB(alpha=1.25))
                      ])
        p.fit(tokens_by_sent, labels)
        return p
    else:
        p = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', SGDClassifier('log'))
                      # ('clf', BernoulliNB(alpha=1.25))
                      ])
        parameters = {'vect__ngram_range':[(1,1),(1,2),(2,1),(2,2)],
                          'tfidf__sublinear_tf':(True, False),
                          'clf__alpha': (.0001, .0005, .005, .00001, .00005),
                          #'clf__alpha':(1.0, 1.25, 1.5, 1.75, 2.0),
                          'clf__penalty': ('elasticnet', 'l2', 'l1'),
                          #'clf__learning_rate': ('constant', 'optimal'), #, 'invscaling'),
                          'clf__warm_start': (True, False),
        }

        gs_clf = GridSearchCV(p, parameters, n_jobs=-1)
        gs_clf = gs_clf.fit(tokens_by_sent, labels)
        print(gs_clf.best_score_)
        for param_name in sorted(parameters.keys()):
            print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
        return gs_clf

def train_by_doc_vec(vecs, labels, dev_vecs=None, dev_labels=None, parameter_search=False):
    if not parameter_search:
        v = SGDClassifier(loss='log', penalty='elasticnet', max_iter=1000, l1_ratio=.65)
        #v = BernoulliNB()
        v.fit(vecs,labels)
        return v
    else:
        split_list = np.repeat([-1, 0], [len(vecs), len(dev_vecs)])
        p = Pipeline([
            ('clf', SGDClassifier(penalty='elasticnet', max_iter=1000),)
        ])
        parameters = {
            #'clf__alpha': (.0001, .0005, .005, .00001, .00005),
            #'clf__loss': ('log', 'perceptron'),
            #'clf__penalty': ('elasticnet', 'l2', 'l1'),
            #'clf__warm_start': (True, False)
        'clf__l1_ratio': (.10, .15, .20, .25, .30, .45, .5, .55, .60, .65, .70, .75)
        }
        '''p = Pipeline([('clf', BernoulliNB())])
        parameters = {
            'clf__alpha': (1.0, 1.25, 1.50, 1.75, 2.0)
        }'''
        v = list()
        l = list()
        v.extend(vecs)
        v.extend(dev_vecs)
        l.extend(labels)
        l.extend(dev_labels)
        gs_clf = GridSearchCV(p, parameters, scoring='f1', cv=list(PredefinedSplit(test_fold=split_list).split(vecs, labels)), refit=False)
        gs_clf.fit(v, l)
        print(gs_clf.best_score_)
        for param_name in sorted(parameters.keys()):
            print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
        pred = SGDClassifier(penalty='elasticnet', max_iter=1000, l1_ratio=gs_clf.best_params_['clf__l1_ratio'],)
        pred.fit(vecs, labels)
        ps = pred.predict(dev_vecs)
        print(metrics.f1_score(dev_labels, ps, pos_label=1))
        print(metrics.classification_report(dev_labels, ps))
        return pred


if __name__ == '__main__':
    main()
