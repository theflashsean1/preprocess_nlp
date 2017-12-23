import tensorflow as tf 
from tensorflow.python.ops import lookup_ops
from vocab_utils.common import UNK_ID, UNK

class VocabReader(object):
    def __init__(self, vocab_f_path, count_f_path=None):
        self._vocab_f_path = vocab_f_path
        self._count_f_path = count_f_path
        self._vocab_size = None
        self._vocab_counts = None
        self._word2id_table = None
        self._id2word_table = None

    def id2word_lookup(self, id_token):
        if not self._id2word_table:
            with tf.variable_scope("vocab_lookup"):
                self._id2word_table = lookup_ops.index_to_string_table_from_file(
                    vocab_f_path, default_value=UNK, name="id2word"
                )
        return self._id2word_table

    def word2id_lookup(self, word_token):
        if not self._word2id_table:
            with tf.variable_scope("vocab_lookup"):
                self._word2id_table = lookup_ops.index_table_from_file(
                    vocab_f_path, default_value=UNK_ID, name="word2id"
                )
        return self._word2id_table

    @property
    def vocab_size(self):
        if not self._vocab_size:
            with open(self._vocab_f_path) as f:
                self._vocab_size = len(f.read().strip().split())
        return self._vocab_size

    @property
    def vocab_counts_list(self):
        if not self._vocab_counts and count_f_path:
            with open(count_f_path) as count_f:
                counts = count_f.read().strip().split()
                self._vocab_counts = [int(count) for count in counts]
        return self._vocab_counts



