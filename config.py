"""Allows for creation of Config objects which store file locations and parameters for different models."""
from pathlib import Path
from spacy import load

from utils import check_dir, prevent_sentence_boundary_detection


def get_specifics(section, generics):
    return [section / generic for generic in generics]


class Config:
    """Contains data locations, where to save files, and parameters for the learning models of tasks 1 and 2"""
    def __init__(self, load_models=False):
        for directory in [self.compressed, self.train, self.dev, self.test, self.test_comb]:
            check_dir(directory)
        if load_models:
            self.vec_model = load('en_core_web_lg')
            self.feat_model = load('en_core_web_lg')
            self.feat_model.add_pipe(prevent_sentence_boundary_detection, name='prevent-sbd', before='parser')

    # source data locations
    train_data_folder = Path('training_material') / 'data' / 'tokenized'
    dev_data_folder = Path('dev') / 'dev_source' / 'tokenized'
    test_data_folder = Path('gold') / 'gold_source' / 'test_1' / 'tokenized'

    # task in and out file locations
    test1_in = Path('eval') / 'Subtask1_2.in'
    task1_out = 'Task1.out'
    task2_out = 'Task2.out'

    # compressed directories
    compressed = Path('compressed')
    train = compressed / 'train'
    dev = compressed / 'dev'
    test = compressed / 'test'
    test_comb = compressed / 'test_comb'

    # generic compressed file names
    tokens_by_sent = 'tokens_by_sent.msgpack'
    labels_by_sent = 'labels_by_sent.msgpack'
    sent_text = 'sent_text.msgpack'
    sent_labels = 'sent_labels.msgpack'
    sent_vec = 'sent_vec.msgpack'
    sent_predict = 'sent_predict.msgpack'
    feat = 'feat.msgpack'
    vec = 'vec.msgpack'
    generics = [tokens_by_sent, labels_by_sent, sent_text, sent_labels, sent_vec, sent_predict, feat, vec]

    # train
    train_tokens_by_sent, train_labels_by_sent, train_sent_text, train_sent_labels, train_sent_vec, \
        train_predict, train_feat, train_vec = get_specifics(train, generics)
    # dev
    dev_tokens_by_sent, dev_labels_by_sent, dev_sent_text, dev_sent_labels, dev_sent_vec, \
        dev_predict, dev_feat, dev_vec = get_specifics(dev, generics)
    # test
    test_tokens_by_sent, test_labels_by_sent, test_sent_text, test_sent_labels, test_sent_vec, \
        test_predict, test_feat, test_vec = get_specifics(test, generics)

    # data from the combined file
    task_1_comb = test_comb / 'task_1.msgpack'
    task_1_comb_sent_vec = test_comb / 'task_1_sent_vecs.msgpack'
    task_2_comb_tokens = test_comb / 'task_2_tokens.msgpack'
    task_2_comb_feat = test_comb / 'task_2_feat.msgpack'
    task_2_comb_vec = test_comb / 'task_2_vec.msgpack'
    test_comb_predict = test_comb / 'sent_predict.msgpack'

    # general parameters
    verbose = True
    write_test_results = False
    use_predefined_split = True
    cv_size = 5

    # task 1 parameters
    sent_text_parameter_search = False
    sent_vec_parameter_search = False
    sent_combine_models = False
    sent_clf_weights = [1.66, 0.33]
    find_weights = True
    range = [0., 1.]
    num_steps = 11
    write_sent_prob = False

    # task 2 parameters
    crf_parameter_search = False
    use_word_vec = False
    use_sentence_prediction = True
