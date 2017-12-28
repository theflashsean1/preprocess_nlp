import tensorflow as tf
from preprocess_nlp.corpus_utils.interface import DocumentState


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_features(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def _bytes_feature_list(values, byte_feature_func):
    return tf.train.FeatureList(feature=[byte_feature_func(val) for val in values])


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_features(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _int64_feature_list(values, int_feautre_func):
    return tf.train.FeatureList(feature=[int_feautre_func(val) for val in values])

def _feature_dict(int_feature_dict, bytes_feature_dict):
    feature_dict = {}
    for key, val in int_feature_dict.items():
        feature_dict[key] = _int64_feature(val)
    for key, val in bytes_feature_dict.items():
        feature_dict[key] = _bytes_feature(val)
    return feature_dict

def _feature_list_dict(int_feature_dict, bytes_feature_dict):
    feature_dict = {}
    for key, val in int_feature_dict.items():
        feature_dict[key] = _int64_feature_list(val, _int64_feature)
    for key, val in bytes_feature_dict.items():
        feature_dict[key] = _bytes_feature_list(val, _bytes_feature)
    return feature_dict

def make_sequence_example(context_feature_dict, feature_list_dict):
    ex = tf.train.SequenceExample(
            context=tf.train.Features(
                feature=context_feature_dict
            ),
            feature_lists=tf.train.FeatureLists(
                feature_list=feature_list_dict    
            )
         )
    return ex

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

    def doc_save_with_label(self, doc_gen, doc_path, doc_path_sub=None):
        """
        doc_gen could be (src, label_dict) or (src, tgt, label_dict)
        """
        pass



