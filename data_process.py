import os
from collections import defaultdict
import json


def process_tokens(token_folder_name):
    tokens_by_doc = defaultdict(list)
    tokens_by_sent = list()
    labels_by_sent = list()
    for file_ in os.listdir(token_folder_name):
        with open(os.path.join(token_folder_name, file_), encoding='utf-8') as f:
            tokens_by_doc[file_[:-7]] = list()
            f = f.readlines()
            sent = list()
            for line in f:
                if line.split():
                    sent.append(line.split())
                else:
                    tokens_by_doc[file_[:-7]].append(sent)
                    tokens_by_sent.append([s[0] for s in sent])
                    labels_by_sent.append([s[-1] for s in sent])
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

def main():
    folder_name = os.path.join('training_material', 'data')
    token_folder_name = os.path.join(folder_name, 'tokenized')
    annotations_folder_name = os.path.join(folder_name, 'annotations')
    plaintext_folder_name = os.path.join(folder_name, 'plaintext')
    tokens = process_tokens(token_folder_name)
    '''with open('training_tokens.json', 'w') as token_file:
        token_file.write(json.dump(tokens, token_file))'''
    # annotations = process_annotations(annotations_folder_name)
    # plaintext = process_plaintext(plaintext_folder_name)


if __name__ == '__main__':
    main()
