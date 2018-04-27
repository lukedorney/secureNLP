from data_process import process_tokens
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.base import TransformerMixin, BaseEstimator

import msgpack

def main():
    folder_name = Path('training_material') / 'data'
    token_folder_name = Path(folder_name) / 'tokenized'
    _, tokens_by_sent, labels = process_tokens(token_folder_name, sent_classification=True)

    folder_name = Path('dev') / 'dev_source'
    token_folder_name = Path(folder_name) / 'tokenized'
    _, dev_tokens_by_sent, dev_labels = process_tokens(token_folder_name, sent_classification=True)

    with open('vec_sent_vec_train_output.msgpack', 'rb') as vec_training:
        vecs = msgpack.unpack(vec_training)
    vecs = [[float(v) for v in vec.split()] for vec in vecs]

    with open('vec_sent_vec_dev_output.msgpack', 'rb') as vec_dev:
        dev_vecs = msgpack.unpack(vec_dev)
    dev_vecs = [[float(v) for v in vec.split()] for vec in dev_vecs]

    # p = train_sent_classifier(tokens_by_sent, labels)
    v = train_by_doc_vec(vecs, labels)

    '''eclf = VotingClassifier(estimators=[('words', p), ('vecs', v)], voting='hard')
    for clf, label in zip([p, v, eclf], ['Logistic Regression', 'Random Forest', 'Ensemble']):
        scores = cross_val_score(clf, dev_vecs, dev_labels, cv=5, scoring='f1')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))'''

    y_pred = v.predict(dev_vecs)
    # y_pred = p.predict(dev_tokens_by_sent)
    acc = np.mean(y_pred == dev_labels)
    print((str(acc)))
    print(metrics.f1_score(dev_labels,y_pred))
    print(metrics.classification_report(dev_labels, y_pred))


def train_sent_classifier(tokens_by_sent, labels):
    p = Pipeline([('vect', CountVectorizer(ngram_range=(1, 1))),
                  ('tfidf', TfidfTransformer(sublinear_tf=True)),
                  ('clf', SGDClassifier(warm_start=False, loss='log', penalty='l2'))
                  # ('clf', BernoulliNB(alpha=1.25))
                  ])
    '''
    parameters = {'vect__ngram_range':[(1,1),(1,2),(2,1),(2,2)],
                      'tfidf__sublinear_tf':(True, False),
                      'clf__alpha':(1.0, 1.25, 1.5, 1.75, 2.0)
                      #'clf__loss': ('log','hinge', 'squared_hinge', 'perceptron'),
                      #'clf__penalty': ('elasticnet', 'l2', 'l1'),
                      #'clf__learning_rate': ('constant', 'optimal', 'invscaling'),
                      #'clf__warm_start': (True, False),
    }

    gs_clf = GridSearchCV(p, parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(tokens_by_sent, labels)
    print(gs_clf.best_score_)
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
    '''
    p.fit(tokens_by_sent, labels)
    return p

def train_by_doc_vec(vecs, labels):
    #v = SGDClassifier(loss='log', penalty='l2')
    v = BernoulliNB()
    v.fit(vecs,labels)
    return v


if __name__ == '__main__':
    main()
