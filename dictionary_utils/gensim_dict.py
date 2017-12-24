from gensim import corpora
from preprocess_nlp import file_utils


def create_dictionary(*text_paths):
    with file_utils.common.AggregatedReadOpen(*text_paths) as read_f:
        dictionary = corpora.Dictionary(line.split() for line in read_f)
    return dictionary


def save_dictionary(dictionary, dict_path):
    dictionary.save(dict_path)


def load_dictionary(dict_path):
    return corpora.Dictionary.load(dict_path)
