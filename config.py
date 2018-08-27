"""Allows for creation of Config objects which store file locations and parameters for different models."""
from pathlib import Path
from spacy import load


class Config:

    def __init__(self, load_models=False):
        if load_models:
            self.vec_model = load('en_core_web_lg')
            self.feat_model = load('en_core_web_lg', disable=['ner'])
            self.feat_model.add_pipe(self.prevent_sentence_boundary_detection, name='prevent-sbd', before='parser')

    def prevent_sentence_boundary_detection(doc):
        """Function to keep sent boundaries found in data for spacy model.

        :param doc: spacy doc made up of processed tokens
        """
        for token in doc:
            token.is_sent_start = False
        return doc

    verbosity = True

    training_data_folder = Path('training_material') / 'data' / 'tokenized'
    dev_data_folder = Path('dev') / 'dev_source' / 'tokenized'
    test_data_folder = Path('gold') / 'gold_source' / 'test_1' / 'tokenized'

    training_sent_vecs = 'sent_vec_train.msgpack'
    dev_sent_vecs = 'sent_vec_dev.msgpack'
    test_sent_vecs = 'sent_vec_test1.msgpack'

    training_predict = 'train_sent_predict.msgpack'
    dev_predict = 'dev_sent_predict.msgpack'
    test_predict = 'test_sent_predict.msgpack'

    training_feat = 'feat_train.msgpack'
    training_vecs = 'vec_train.msgpack'
    dev_feat = 'feat_dev.msgpack'
    dev_vecs = 'vec_dev.msgpack'
    test_feat = 'feat_test1.msgpack'
    test_vecs = 'vec_test1.msgpack'

    sent_text_parameter_search = False
    sent_vec_parameter_search = False
    sent_combine_models = False
    sent_clf_weights = [1.66, 0.33]
    find_weights = True
    range = [0., 1.]
    num_steps = 11
    write_probs = True

    crf_parameter_search = False
    use_word_vec = True
