from gensim import corpora
import preprocess_nlp.file_utils as fu


def create_dictionary(*text_paths):
    with fu.common.AggregatedReadOpen(*text_paths) as read_f:
        dictionary = corpora.Dictionary(line.split() for line in read_f)
    return dictionary


def save_dictionary(dictionary, dict_path):
    dictionary.save(dict_path)


def load_dictionary(dict_path):
    return corpora.Dictionary.load(dict_path)
