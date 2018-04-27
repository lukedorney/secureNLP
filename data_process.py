import os
from collections import defaultdict
from spacy import tokens, load
import umsgpack
from numpy.random import uniform
from nltk.tokenize.moses import MosesDetokenizer


def process_tokens(token_folder_name, sent_classification=False):
    if sent_classification:
        m = MosesDetokenizer()
    tokens_by_doc = defaultdict(list)
    tokens_by_sent = list()
    labels_by_sent = list()
    for file_ in os.listdir(token_folder_name):
        with open(os.path.join(token_folder_name, file_), 'r', encoding='utf-8') as f:
            tokens_by_doc[file_[:-7]] = list()
            f = f.readlines()
            sent = list()
            if not sent_classification:
                for line in f:
                    if line.split():
                        sent.append(line.split())
                    else:
                        tokens_by_doc[file_[:-7]].append(sent)
                        tokens_by_sent.append([s[0] for s in sent])
                        labels_by_sent.append([s[-1] for s in sent])
                        sent = list()
            else:
                sent_labels = list()
                for line in f:
                    if line.split():
                        sent.append(line.split())
                    else:
                        tokens_by_doc[file_[:-7]].append(sent)
                        tokens_by_sent.append(m.detokenize(tokens=[s[0] for s in sent], return_str=True))
                        labels_by_sent.append(0 if all(label[-1] == 'O' for label in sent) else 1)
                        sent = list()
    return tokens_by_doc, tokens_by_sent, labels_by_sent


def process_annotations(annotations_folder_name):
    annotations = dict()
    for file_ in [f for f in os.listdir(annotations_folder_name) if f.endswith('.ann')]:
        with open(os.path.join(annotations_folder_name, file_), encoding='utf-8') as f:
            file_ = file_[:-4]
            annotations[file_] = defaultdict(dict)
            f = f.readlines()
            for line in f:
                line = line.split()
                if line:
                    if line[0].startswith('T'):
                        line = line[:4] + [' '.join(line[4:])]
                        annotations[file_]['text'][line[0]] = line[1:]
                    elif line[0].startswith('A'):
                        annotations[file_]['attribute'][line[0]] = line[1:]
                    elif line[0].startswith('R'):
                        annotations[file_]['relation'][line[0]] = line[1:]
    return annotations


def process_plaintext(plaintext_folder_name):
    plaintext = dict()
    for file_ in os.listdir(plaintext_folder_name):
        with open(os.path.join(plaintext_folder_name, file_), encoding='utf-8') as f:
            f = f.read()
            plaintext[file_[:-8]] = f
    return plaintext


def prevent_sentence_boundary_detection(doc):
    for token in doc:
        token.is_sent_start = False
    return doc


def add_features(tokens_by_sent, model_name, out_name):
    nlp = load(model_name, disable=['ner'])
    nlp.add_pipe(prevent_sentence_boundary_detection, name='prevent-sbd', before='parser')

    with open(out_name, 'wb') as f:
        sents = []
        for s in tokens_by_sent:
            sent = tokens.Doc(nlp.vocab, words=s)
            nlp.tagger(sent)
            nlp.parser(sent)
            sents.append(
                ["\t".join([token.text, token.pos_, token.tag_, token.dep_, str(token.vector_norm), str(token.cluster),
                 str(token.is_oov), str(token.is_stop), token.head.text, token.head.pos_])
                 for token in sent])
        umsgpack.pack(sents, f)

def add_vecs(tokens_by_sent, model_name, out_name, sent_classification=False):
    nlp = load(model_name)
    vecs = []
    with open('vec_' + out_name, 'wb') as v:
        if not sent_classification:
            for s in tokens_by_sent:
                sent = tokens.Doc(nlp.vocab, words=s)
                vecs.append(['\t'.join([str(val) for val in token.vector]) if not sum(token.vector) == 0.0 else '\t'.join([str(val) for val in uniform(-1, 1, 300)]) for token in sent])
        else:
            for s in tokens_by_sent:
                vecs.append('\t'.join([str(val) for val in nlp(s).vector]))

        umsgpack.pack(vecs, v)

def main():
    folder_name = os.path.join('training_material', 'data')
    token_folder_name = os.path.join(folder_name, 'tokenized')
    tokens_by_doc, tokens_by_sent, labels_by_sent = process_tokens(token_folder_name, sent_classification=True)
    add_vecs(tokens_by_sent, 'en_core_web_lg', 'sent_vec_train_output.msgpack', sent_classification=True)
    #add_features(tokens_by_sent, 'en_core_web_lg', 'train_output.msgpack')
    print('processed training')

    folder_name = os.path.join('dev', 'dev_source')
    token_folder_name = os.path.join(folder_name, 'tokenized')
    tokens_by_doc, tokens_by_sent, labels_by_sent = process_tokens(token_folder_name, sent_classification=True)
    #add_vecs(tokens_by_sent, 'en_core_web_lg', 'sent_vec_dev_output.msgpack', sent_classification=True)
    #add_features(tokens_by_sent, 'en_core_web_lg', 'dev_output.msgpack')
    print('processed dev')

    # annotations_folder_name = os.path.join(folder_name, 'annotations')
    # plaintext_folder_name = os.path.join(folder_name, 'plaintext')
    # annotations = process_annotations(annotations_folder_name)
    # plaintext = process_plaintext(plaintext_folder_name)


if __name__ == '__main__':
    main()
