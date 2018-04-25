from pathlib import Path
from joblib import Parallel, delayed
import plac
from spacy import load, tokens
import msgpack
from data_process import process_tokens


@plac.annotations(
    output_dir=("Output directory", "positional", None, Path),
    model_name=("Model name (needs tagger)", "positional", None, str),
    n_jobs=("Number of workers", "option", "n", int))
def main(output_dir='train_features', model_name='en_core_web_sm', n_jobs=-1):
    nlp = load(model_name, disable=['ner'])
    nlp.add_pipe(prevent_sentence_boundary_detection, name='prevent-sbd', before='parser')

    folder_name = Path('training_material') / 'data'
    token_folder_name = Path(folder_name) / 'tokenized'
    tokens_by_doc, _, _ = process_tokens(token_folder_name)

    if not Path(output_dir).exists():
        Path(output_dir).mkdir()

    executor = Parallel(n_jobs=n_jobs)
    do = delayed(transform_texts)
    tasks = (do(nlp, tokens_by_doc[doc_name], doc_name, output_dir) for i, doc_name in enumerate(tokens_by_doc))
    executor(tasks)


def prevent_sentence_boundary_detection(doc):
    for token in doc:
        token.is_sent_start = False
    return doc


def transform_texts(nlp, texts, file_name, output_dir):
    out_path = Path(output_dir) / ('%s.msgpack' % file_name)
    if out_path.exists():  # return None in case same batch is called again
        return None

    with out_path.open('w', encoding='utf8') as f:
        document = list()
        for sent in nlp.pipe(texts):
            sent = tokens.Doc(nlp.vocab, words=sent)
            nlp.tagger(sent)
            nlp.parser(sent)
            document.append(["\t".join([token.text, token.pos_, token.tag_, token.dep_, str(token.is_stop), token.head.text, token.head.pos_]) for token in sent])
            document.append('\n')
        msgpack.dump(document, f)


if __name__ == '__main__':
    plac.call(main)
