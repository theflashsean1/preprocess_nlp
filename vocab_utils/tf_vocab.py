import tensorflow as tf


class VocabReader(object):
    def __init__(self, vocab_f_path, count_f_path=None):
        pass

    def tf_ids_lookup_op(self, str_tensor):
        return tf.cast(self._word2id_table.lookup(str_tensor), tf.int32)

    def tf_words_lookup_op(self, id_tensor):
        return self._id2word_table.lookup(tf.to_int64(id_tensor))

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def vocab_counts_list(self):
        return self._vocab_counts

