import tensorflow as tf
from functools import partial
from preprocess_nlp.doc_token import WORD_TYPE, ID_TYPE, VALUE_TYPE


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


def _feature_lists_dict(int_feature_dict, bytes_feature_dict):
    feature_dict = {}
    for key, val in int_feature_dict.items():
        feature_dict[key] = _int64_feature_list(val, _int64_feature)
    for key, val in bytes_feature_dict.items():
        feature_dict[key] = _bytes_feature_list(val, _bytes_feature)
    return feature_dict


def make_example(feature_dict):
    ex = tf.train.Example(
            features=tf.train.Features(features=feature_dict)   
         )
    return ex


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


def doc_save(self, doc, doc_transformer, tfrecords_save_path):
    feature_fs = []
    for token_type, seq_len in zip(doc_transformer.token_types, doc_transformer.seq_lens):
        if token_type == WORD_TYPE:
            if seq_len > 1:
                feature_fs.append((partial(_bytes_feature_list, byte_feature_func=_bytes_feature), 1))
            else:
                feature_fs.append((_bytes_feature, 0))
        elif token_type == ID_TYPE:
            if seq_len > 1:
                feature_fs.append((partial(_int64_feature_list, int_feature_func=_int64_feature), 1))
            else:
                feature_fs.append((_int64_feature, 0))

    with tf.python_io.TFRecordWriter(tfrecords_save_path) as writer:
        for seqs in doc_transformer.get_iters(doc):
            context_feature_dict = {doc_transformer.iter_keys[i]:feature_fs[i][0](seq) 
                     for i, seq in  enumerate(seqs) if feature_fs[i][1]==0}
            feature_lists_dict = {doc_transformer.iter_keys[i]:feature_fs[i][0](seq) 
                     for i, seq in  enumerate(seqs) if feature_fs[i][1]==1}
            seq_ex = make_sequence_example(context_feature_dict, feature_lists_dict)
            writer.write(seq_ex.SerializeToString())

"""
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
"""


