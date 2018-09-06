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
import numpy as np
from numpy.random import uniform
from sacremoses import MosesDetokenizer
from spacy import tokens as spacy_tokens

from config import Config
from utils import write_msgpack


CONFIG = Config(load_models=True)


def add_features(tokens_by_sent, output):
    """ Writes token features to compressed file.

    :param tokens_by_sent: list of list of tokens
    :param output: desired output folder name (as a Path object) e.g. train, test, dev
    """
    sentences = _add_features(tokens_by_sent)
    write_msgpack(sentences, output / CONFIG.feat)


def _add_features(tokens_by_sent):
    """Helper function for getting features.

    :param tokens_by_sent: list of list of tokens
    :return: list of list of features
    """
    sentences = []
    for s in tokens_by_sent:
        sent = spacy_tokens.Doc(CONFIG.feat_model.vocab, words=s)
        CONFIG.feat_model.tagger(sent)
        CONFIG.feat_model.parser(sent)
        CONFIG.feat_model.entity(sent)
        sentences.append(['\t'.join([token.text, token.pos_, token.tag_, token.dep_, str(token.vector_norm),
                                     str(token.cluster), str(token.is_oov), str(token.is_stop), token.head.text,
                                     token.head.pos_, token.lemma_, str(token.like_email), str(token.like_url),
                                     str(token.prob), token.ent_iob_, token.ent_type_ if token.ent_type_ else 'NA',
                                     token.shape_])
                          for token in sent])
    return sentences


def add_vec(tokens_by_sent, sent_text, output):
    """Writes word vectors and sent vectors to compressed file.

    :param tokens_by_sent: list of list of tokens
    :param sent_text: list of sentence texts
    :param output: desired output folder name (as a Path object) e.g. train, test, dev
    """
    vecs, sent_vecs = _add_vec(sent_text, tokens_by_sent)
    write_msgpack(vecs, output / CONFIG.vec)
    write_msgpack(sent_vecs, output / CONFIG.sent_vec)


def _add_vec(sent_text, tokens_by_sent):
    """Helper function for getting vectors for each sentence and each word.

    :param sent_text: list of sentences
    :param tokens_by_sent: list of list of sentences
    :return: list of list of vectors, list of sentence vectors
    """
    vec = []
    for s in tokens_by_sent:
        sent = spacy_tokens.Doc(CONFIG.vec_model.vocab, words=s)
        vec.append(['\t'.join(map(str, token.vector))
                    if not np.array_equal(token.vector, np.zeros(300))
                    else '\t'.join(map(str, uniform(-1, 1, 300)))
                    for token in sent])
    sent_vec = ['\t'.join(map(str, CONFIG.vec_model(sent).vector)) for sent in sent_text]
    return vec, sent_vec


def preprocess(folder_name, output):
    """Creates vector and feature files.

    :param folder_name: name of input folder (tokens)
    :param output: desired output folder name (as a Path object) e.g. train, test, dev
    """
    tokens_by_sent, labels_by_sent, sent_text, sent_label = process_tokens(folder_name)
    write_data(tokens_by_sent, labels_by_sent, sent_text, sent_label, output)
    add_features(tokens_by_sent, output)
    add_vec(tokens_by_sent, sent_text, output)


def process_test_combined():
    """Processes test .in file vy reading in data and writing tokens, sents, labels, feats, vecs to compressed file."""
    sentences, tokens = read_test_combined()
    write_test_combined(sentences, tokens)


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


def read_test_combined():
    """Gets the sentences ad tokens for test data that is combined into one file (without label info).

    :return: list of sentences, list of list of tokens
    """
    with open(CONFIG.test1_in, encoding='utf-8') as in_file:
        in_file = in_file.readlines()
    sentences, tokens, sent = list(), list(), list()
    m = MosesDetokenizer()

    for line in in_file:
        line = line.strip()
        if line:
            sent.append(line)
        else:
            tokens.append(sent)
            sentences.append(m.detokenize(tokens=sent, return_str=True))
            sent = list()

    return sentences, tokens


def write_data(tokens_by_sent, labels_by_sent, sent_text, sent_label, output):
    """Write data and labels to file in compressed files.

    :param tokens_by_sent: list of list of tokens,, ,
    :param labels_by_sent:  list of list of labels
    :param sent_text: list of sentence texts
    :param sent_label: list of sentence labels
    :param output: desired output folder name (as a Path object) e.g. train, test, dev
    """
    write_msgpack(tokens_by_sent, output / CONFIG.tokens_by_sent)
    write_msgpack(labels_by_sent, output / CONFIG.labels_by_sent)
    write_msgpack(sent_text, output / CONFIG.sent_text)
    write_msgpack(sent_label, output / CONFIG.sent_labels)


def write_test_combined(sentences, tokens):
    """Writes to file all info processed from combined test .in file.

    :param sentences: list of sentences
    :param tokens: list of list of tokens
    """
    write_msgpack(sentences, CONFIG.task_1_comb)
    write_msgpack(tokens, CONFIG.task_2_comb_tokens)
    write_msgpack(_add_features(tokens), CONFIG.task_2_comb_feat)
    vec, sent_vec = _add_vec(sentences, tokens)
    write_msgpack(vec, CONFIG.task_2_comb_vec)
    write_msgpack(sent_vec, CONFIG.task_1_comb_sent_vec)


def main():
    """Controls all pre-processing work (for train, dev, and test sets)."""

    preprocess(CONFIG.train_data_folder, CONFIG.train)
    if CONFIG.verbose:
        print('processed training')

    preprocess(CONFIG.dev_data_folder, CONFIG.dev)
    if CONFIG.verbose:
        print('processed dev')

    preprocess(CONFIG.test_data_folder, CONFIG.test)
    if CONFIG.verbose:
        print('processed test')

    process_test_combined()
    if CONFIG.verbose:
        print('processed combined test')


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
