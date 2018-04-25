from data_process import process_tokens
import os
import sklearn_crfsuite
# from itertools import chain
# import sklearn
# from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
import msgpack


def main():
    folder_name = os.path.join('training_material', 'data')
    token_folder_name = os.path.join(folder_name, 'tokenized')
    _, _, labels = process_tokens(token_folder_name)
    with open('train_output.msgpack', 'rb') as training:
        tokens = msgpack.unpack(training)
    x_train = (sent2features(s) for s in tokens)
    print('processed train')
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=False
    )
    crf.fit(x_train, labels)
    print('fit data')
    all_labels = list(crf.classes_)
    content_labels = list(crf.classes_)
    content_labels.remove('O')

    folder_name = os.path.join('dev', 'dev_source')
    token_folder_name = os.path.join(folder_name, 'tokenized')
    _, _, dev_labels = process_tokens(token_folder_name)
    with open('dev_output.msgpack', 'rb') as dev:
        dev_tokens = msgpack.unpack(dev)
    dev_x_train = (sent2features(s) for s in dev_tokens)
    print('processed dev')

    y_pred = crf.predict(dev_x_train)
    results = metrics.flat_f1_score(dev_labels, y_pred,
                                    average='weighted', labels=all_labels)
    print('dev results with Os -- ' + str(results))
    results = metrics.flat_f1_score(dev_labels, y_pred,
                                    average='weighted', labels=content_labels)
    print('dev results without Os -- ' + str(results))

    sorted_labels = sorted(
        content_labels,
        key=lambda name: (name[1:], name[0])
    )
    print(metrics.flat_classification_report(
        dev_labels, y_pred, labels=sorted_labels, digits=3
    ))



def featurize(word_index, word, sent):
    # 0 token.text, 1 token.pos_, 2 token.tag_, 3 token.dep_, 4 str(token.vector_norm), 5 str(token.cluster),
    #  6 str(token.is_oov), 7 str(token.is_stop), 8 token.head.text, 9 token.head.pos_
    word = word.split()

    features = {'bias': 1.0,
                'word_lower': word[0].lower(),
                'word[-3:]': word[0][-3:],
                'word[-2:]': word[0][-2:],
                'word.isupper()': word[0].isupper(),
                'word.istitle()': word[0].istitle(),
                'word.isdigit()': word[0].isdigit(),
                'pos': word[1],
                'tag': word[2],
                'dep': word[3],
                'vector_norm': float(word[4]),
                'cluster': word[5],
                'oov': bool(word[6]),
                'is_stop': bool(word[7]),
                'dep_head': word[8],
                'dep_head_pos': word[9]}

    if word_index > 0:
        prev_word = sent[word_index - 1].split()
        features.update({
            'prev_word.lower()': prev_word[0].lower(),
            'prev_word.istitle()': prev_word[0].istitle(),
            'prev_word.isupper()': prev_word[0].isupper(),
            'prev_word.isdigit()': prev_word[0].isdigit(),
            'prev_word_pos': prev_word[1],
            'prev_word_tag': prev_word[2],
            'prev_word_dep': prev_word[3],
            'prev_word_vector_norm': float(prev_word[4]),
            'prev_word_cluster': prev_word[5],
            'prev_word_oov': bool(prev_word[6]),
            'prev_word_is_stop': bool(prev_word[7])})
    else:
        features['BOS'] = True

    if word_index < len(sent) - 1:
        next_word = sent[word_index + 1].split()
        features.update({
            'next_word.lower()': next_word[0].lower(),
            'next_word.istitle()': next_word[0].istitle(),
            'next_word.isupper()': next_word[0].isupper(),
            'next_word.isdigit()': next_word[0].isdigit(),
            'next_word_pos': next_word[1],
            'next_word_tag': next_word[2],
            'next_word_dep': next_word[3],
            'next_word_vector_norm': float(next_word[4]),
            'next_word_cluster': next_word[5],
            'next_word_oov': bool(next_word[6]),
            'next_word_is_stop': bool(next_word[7])
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return (featurize(i, word, sent) for i, word in enumerate(sent))


if __name__ == '__main__':
    main()
