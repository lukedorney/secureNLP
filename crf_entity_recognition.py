from data_process import process_tokens
import os
import sklearn_crfsuite
from itertools import chain
import sklearn
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

def main():
    folder_name = os.path.join('training_material', 'data')
    token_folder_name = os.path.join(folder_name, 'tokenized')
    sentences_by_doc, tokens, labels = process_tokens(token_folder_name)
    x_train = (sent2features(s) for s in tokens[:2])
    print('processed data')
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(x_train, labels)
    print('fit data')
    all_labels = list(crf.classes_)
    all_labels.remove('O')

    folder_name = os.path.join('dev_material', 'dev_data')
    token_folder_name = os.path.join(folder_name, 'tokenized')
    dev_sentences_by_doc, dev_tokens, dev_labels = process_tokens(token_folder_name)
    dev_x_train = (sent2features(s) for s in dev_tokens[:2])
    print('processed dev')
    y_pred = crf.predict(dev_x_train)
    results = metrics.flat_f1_score(dev_labels, y_pred,
                          average='weighted', labels=all_labels)
    print('dev results' + str(results))


def featurize(word_index, word, sent):

    features = {'bias': 1.0,
        'word.lower()': word.lower(),
        #'word[-3:]': word[-3:],
        #'word[-2:]': word[-2:],
        #'word.isupper()': word.isupper(),
        #'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit()}
    if word_index > 0:
        word1 = sent[word_index - 1][0]
        features.update({
            '-1:word.lower()': word1.lower(),
            #'-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper()})
    else:
        features['BOS'] = True

    if word_index < len(sent) - 1:
        word1 = sent[word_index + 1][0]
        features.update({
            '+1:word.lower()': word1.lower(),
            #'+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True

    return features

def sent2features(sent):
    yield (featurize(i, word, sent) for i, word in enumerate(sent))




if __name__ == '__main__':
    main()