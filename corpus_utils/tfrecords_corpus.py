import tensorflow as tf
from preprocess_nlp.corpus_utils.interface import DocumentState


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_features(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_features(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


# TODO not needed here for recovering data from .tfrecords
def get_parse_func(src_type, tgt_type):
    def _parse(example_proto):
        features = tf.parse_single_example(
            serialized=example_proto,
            features={
                'src': tf.FixedLenFeature([], src_type),
                'tgt': tf.FixedLenFeature([], tgt_type)
            }
        )
        return features['src'], features['tgt']
    return _parse

# Put this in tensorbridge models
def get_parse_func2(src_type, tgt_type, batch_size):
    def _parse(example_proto):
        features = tf.parse_single_example(
            serialized=example_proto,
            features={
                'src_seq': tf.FixedLenFeature([], src_type),   
                'label': tf.FixedLenFeature([], tgt_type),
                'flag_new_doc': tf.FixedLenFeature([], tf.int32)
            }
        )


class TfrecordsDocumentState(DocumentState):
    @property
    def doc_format(self):
        return "tfrecords"

    def doc_gen_func(self, doc_path):
        def doc_gen():
            pass
        return doc_gen

    def doc_save(self, doc_gen, doc_path, doc_path_sub=None):
        if self._token_type == str:
            feature_func = _bytes_feature
        items = next(doc_gen)
        if len(items) == 2:
            pass
        elif len(items) == 1:
            pass
        else:
            raise NotImplementedError(str(len(items)) + " gen items not handled right now")


def gen_tfrecords_doc(num_examples, **kwdocs):
    for _ in num_examples:
        context_feature_dict = {}
        feature_lists_dict = {}


