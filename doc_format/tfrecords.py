import tensorflow as tf
import pdb
from functools import partial
from preprocess_nlp.doc_token import WORD_TYPE, ID_TYPE, VALUE_INT_TYPE, VALUE_FLOAT_TYPE 
from preprocess_nlp.file_utils.common import limit_iter


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def _bytes_features(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode() for value in values]))


def _bytes_feature_list(values, byte_feature_func):
    return tf.train.FeatureList(feature=[byte_feature_func(val) for val in values])


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_features(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _int64_feature_list(values, int_feautre_func):
    return tf.train.FeatureList(feature=[int_feautre_func(val) for val in values])


def _float64_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _float64_features(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def _float_feature_list(values, float_feature_func):
    return tf.train.FeatureList(feature=[float_feature_func(val) for val in values])


def _feature_dict(int_feature_dict, bytes_feature_dict, float_feature_dict):
    feature_dict = {}
    for key, val in int_feature_dict.items():
        feature_dict[key] = _int64_feature(val)
    for key, val in bytes_feature_dict.items():
        feature_dict[key] = _bytes_feature(val)
    for key, val in float_feature_dict.items():
        feature_dict[key] = _float64_feature(val)
    return feature_dict


def _feature_lists_dict(int_feature_dict, bytes_feature_dict, float_feature_dict):
    feature_dict = {}
    for key, val in int_feature_dict.items():
        feature_dict[key] = _int64_feature_list(val, _int64_feature)
    for key, val in bytes_feature_dict.items():
        feature_dict[key] = _bytes_feature_list(val, _bytes_feature)
    for key, val in float_feature_dict.items():
        feature_dict[key] = _float_feature_list(val, _float64_feature)
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


def doc_save(doc_transform_state, tfrecords_save_path):
    feature_fs = []
    doc_transformer = doc_transform_state.transformer
    for token_type, seq_len in zip(doc_transformer.token_types, doc_transformer.seq_lens):
        if token_type == WORD_TYPE:
            if seq_len > 1:
                feature_fs.append((partial(_bytes_feature_list, byte_feature_func=_bytes_feature), 1))
            else:
                feature_fs.append((_bytes_feature, 0))
        elif token_type == ID_TYPE or token_type == VALUE_INT_TYPE:
            if seq_len > 1:
                feature_fs.append((partial(_int64_feature_list, int_feature_func=_int64_feature), 1))
            else:
                feature_fs.append((_int64_feature, 0))
        elif token_type == VALUE_FLOAT_TYPE:
            if seq_len > 1:
                feature_fs.append((partial(_float_feature_list, float_feature_func= _float64_feature), 1))
            else:
                feature_fs.append((_float64_feature, 0))
        else:
            raise ValueError("Not valid token type")
    try:
        get_iters_f = doc_transformer.get_better_iters
    except AttributeError:
        get_iters_f = doc_transformer.get_iters
    except:
        raise ValueError("No iter method implemented on this doc transformer")
        
    with tf.python_io.TFRecordWriter(tfrecords_save_path) as writer:
        seqs_iter = get_iters_f(*doc_transform_state.docs)
        for seqs in limit_iter(seqs_iter, doc_transform_state.size):
            # pdb.set_trace()
            context_feature_dict = {doc_transformer.iter_keys[i]:feature_fs[i][0](seq) 
                     for i, seq in  enumerate(seqs) if feature_fs[i][1]==0}
            feature_lists_dict = {doc_transformer.iter_keys[i]:feature_fs[i][0](seq) 
                     for i, seq in  enumerate(seqs) if feature_fs[i][1]==1}
            seq_ex = make_sequence_example(context_feature_dict, feature_lists_dict)
            writer.write(seq_ex.SerializeToString())


def word2vec_iter_tensors(dataset_path, batch_size, vocab_reader=None):
    def _parse(example_proto):
        features=tf.parse_single_example(
                    serialized=example_proto,
                    features={
                        'center': tf.FixedLenFeature([], tf.string),
                        'context': tf.FixedLenFeature([], tf.string)
                    }
                )
        return features['center'], features['context'] 
    dataset = tf.data.TFRecordDataset([dataset_path])
    dataset = dataset.map(_parse)
    if vocab_reader:
        dataset = dataset.map(
                    lambda src, tgt: (
                        vocab_reader.word2id_lookup(src),
                        vocab_reader.word2id_lookup(tgt)
                    )
                )
    # dataset = dataset.batch(batch_size)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    iterator = dataset.make_initializable_iterator()
    center, context = iterator.get_next()
    return iterator.initializer, center, context


def word2vec_iter(dataset_path, batch_size):
    initializer, center, context = word2vec_iter_tensors(dataset_path, batch_size)
    with tf.Session() as sess:
        sess.run(initializer)
        while True:
            try:
                center, context = sess.run([center, context])
                yield center.decode(), context.decode()
            except tf.errors.OutOfRangeError:
                break


def sca2word_iter_tensors(dataset_path, batch_size, vocab_reader=None):
    def _parse(example_proto):
        features=tf.parse_single_example(
                    serialized=example_proto,
                    features={
                        'u_i_token': tf.FixedLenFeature([], tf.string),
                        'w_i': tf.FixedLenFeature([], tf.int64),
                        'v_i_token': tf.FixedLenFeature([], tf.string),
                        'u_j_token': tf.FixedLenFeature([], tf.string),
                        'w_j': tf.FixedLenFeature([], tf.int64),
                        'v_j_token': tf.FixedLenFeature([], tf.string)
                    }
                )
        return features['u_i_token'], features['w_i'], features['v_i_token'], features['u_j_token'], features['w_j'], features['v_j_token']
    dataset = tf.data.TFRecordDataset([dataset_path])
    dataset = dataset.map(_parse)
    if vocab_reader:
        dataset = dataset.map(
                    lambda u_i, w_i, v_i, u_j, w_j, v_j:(
                        vocab_reader.word2id_lookup(u_i),
                        w_i,
                        vocab_reader.word2id_lookup(v_i),
                        vocab_reader.word2id_lookup(u_j),
                        w_j,
                        vocab_reader.word2id_lookup(v_j)
                    )
                )
    # dataset = dataset.shuffle(8000)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    iterator = dataset.make_initializable_iterator()
    u_i, w_i, v_i, u_j, w_j, v_j = iterator.get_next()
    return iterator.initializer, u_i, w_i, v_i, u_j, w_j, v_j


def sca2scapair_iter_tensors(dataset_path, batch_size):
    def _parse(example_proto):
        features=tf.parse_single_example(
                serialized=example_proto,
                features={
                    'w_i': tf.FixedLenFeature([], tf.int64),
                    'w_j': tf.FixedLenFeature([], tf.int64)
                }
            )
        return features['w_i'], features['w_j']
    dataset = tf.data.TFRecordDataset([dataset_path])
    dataset = dataset.map(_parse)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    iterator = dataset.make_initializable_iterator()
    w_i, w_j = iterator.get_next()
    return iterator.initializer, w_i, w_j

