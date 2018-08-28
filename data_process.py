"""Prepares SecureNLP data for ML.

Expects data in the following BIO format (with newlines between sentences):
    The O
    virus B-Entity
    was O
    deleted O

Creates compressed files containing information about sentences' tokens and vectors:
    feat_XXX.msgpack: linguistic features
    vec_XXX.msgpack: word-level vectors
    sent_vec_XXX.msgpack: sentence-level vectors
"""
import os

import msgpack
import numpy as np
from numpy.random import uniform
from sacremoses import MosesDetokenizer
from spacy import tokens

from config import Config


def add_features(tokens_by_sent, model, data_section):
    """ Writes token features to compressed file.

    :param tokens_by_sent: list of list of tokens
    :param model: spacy model to find features
    :param data_section: desired output name e.g. train, test, dev
    """
    with open('feat_{}.msgpack'.format(data_section), 'wb') as f:
        sents = []
        for s in tokens_by_sent:
            sent = tokens.Doc(model.vocab, words=s)
            model.tagger(sent)
            model.parser(sent)
            # print(list(sent.noun_chunks))
            sents.append(['\t'.join([token.text, token.pos_, token.tag_, token.dep_, str(token.vector_norm),
                                     str(token.cluster), str(token.is_oov), str(token.is_stop), token.head.text,
                                     token.head.pos_, token.lemma_, str(token.like_email), str(token.like_url)])
                          for token in sent])
        msgpack.pack(sents, f)


def add_vecs(tokens_by_sent, sent_text, model, output):
    """Writes word vectors and sent vectors to compressed file.

    :param tokens_by_sent: list of list of tokens
    :param sent_text: list of sentence texts
    :param model: spacy model to find vectors
    :param output: desired output name e.g. train, test, dev
    """
    with open('vec_{}.msgpack'.format(output), 'wb') as v:
        vecs = []
        for s in tokens_by_sent:
            sent = tokens.Doc(model.vocab, words=s)
            vecs.append(['\t'.join(map(str, token.vector))
                         if not np.array_equal(token.vector, np.zeros(300))
                         else '\t'.join(map(str, uniform(-1, 1, 300)))
                         for token in sent])
        msgpack.pack(vecs, v)
    with open('sent_vec_{}.msgpack'.format(output), 'wb') as v:
        sent_vecs = ['\t'.join(map(str, model(sent).vector)) for sent in sent_text]
        msgpack.pack(sent_vecs, v)


def preprocess(folder_name, output, config):
    """Creates vector and feature files.

    :param folder_name: name of input folder (tokens)
    :param output: desired output name e.g. train, test, dev
    :param config: object containing data locations
    """
    tokens_by_sent, labels_by_sent, sent_text, sent_label = process_tokens(folder_name)
    add_features(tokens_by_sent, config.feat_model, output)
    add_vecs(tokens_by_sent, sent_text, config.vec_model, output)


def process_tokens(token_folder_name):
    """Processes data, outputs tokens and labels.

    :param token_folder_name: name of the directory containing tokenized data
    :return: list of list of tokens, list of list of labels, list of sentence texts, list of sentence labels
    """
    m = MosesDetokenizer()
    tokens_by_sent, labels_by_sent, sent_text, sent_label = list(), list(), list(), list()
    for file_ in os.listdir(token_folder_name):
        with open(os.path.join(token_folder_name, file_), 'r', encoding='utf-8') as f:
            f = f.readlines()
            sent = list()
            for line in f:
                line = line.split()
                if line:
                    sent.append(line)
                else:
                    words, labels = zip(*sent)
                    detokenized_sent = m.detokenize(tokens=words, return_str=True)
                    cum_label = 0 if all(label == 'O' for label in labels) else 1

                    tokens_by_sent.append(words)
                    labels_by_sent.append(labels)
                    sent_text.append(detokenized_sent)
                    sent_label.append(cum_label)
                    sent = list()
    return tokens_by_sent, labels_by_sent, sent_text, sent_label


def main():
    """Controls all pre-processing work (for train, dev, and test sets)."""

    c = Config(load_models=True)

    preprocess(c.training_data_folder, 'train', c)
    if c.verbosity:
        print('processed training')

    preprocess(c.dev_data_folder, 'dev', c)
    if c.verbosity:
        print('processed dev')

    preprocess(c.test_data_folder, 'test1', c)
    if c.verbosity:
        print('processed test')


if __name__ == '__main__':
    main()

""" functions for processing rel/attrib and plaintext files

def process_annotations(annotations_folder_name):
    annotations = dict()
    for file_ in [f for f in os.listdir(annotations_folder_name) if f.endswith('.ann')]:
        with open(os.path.join(annotations_folder_name, file_), encoding='utf-8') as f:
            file_ = file_[:-4]
            annotations[file_] = defaultdict(dict)
            f = f.readlines()
            for line in f:
                if line.startswith('T'):
                    line = line.split('\t')
                    tag_number = line[0]
                    text = line[-1][:-1]
                    tag = line[1].split()[0]
                    if ';' in line[1]:
                        indexes = re.split('[ ;]', line[1])[1:]
                        indexes = list(zip(*[iter([int(e) for e in indexes])] * 2))
                    else:
                        line = line[1].split()[1:]
                        indexes = [(int(line[0]), int(line[1]))]
                    annotations[file_]['text'][tag_number] = (tag, indexes, text)
                elif line.startswith('R'):
                    line = line.split()
                    arg1, arg2 = tuple(line[2].split(':')), tuple(line[3].split(':'))
                    annotations[file_]['relation'][line[0]] = [line[1], arg1, arg2]

                    '''elif line.startswith('A'):
                        pass
                        # annotations[file_]['attribute'][line[0]] = line[1:]'''
    return annotations


def process_annotation_text(annotations_folder_name):
    annotation_texts = dict()
    for file_ in [f for f in os.listdir(annotations_folder_name) if f.endswith('.txt')]:
        with open(os.path.join(annotations_folder_name, file_), encoding='utf-8') as f:
            f = f.read()
            annotation_texts[file_[:-4]] = f
            '''
                        parser = ET.XMLParser(recover=True)
                        tree = ET.fromstring(f, parser=parser)
                        print('yay')'''
    return annotation_texts


def process_plaintext(plaintext_folder_name):
    plaintext = dict()
    for file_ in os.listdir(plaintext_folder_name):
        with open(os.path.join(plaintext_folder_name, file_), encoding='utf-8') as f:
            f = f.read()
            plaintext[file_[:-8]] = f
    return plaintext
"""
