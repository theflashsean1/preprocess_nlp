WORD_TYPE = "word_type"  # str
ID_TYPE = "id_type"      # int
VALUE_INT_TYPE = "value_int_type"
VALUE_FLOAT_TYPE = "value_float_type"
SEQ_TYPE = "seq_type"

# SEQ_WORDS = "seq_words"
# SEQ_IDS = "seq_ids"
# SEQ_EMBEDS = "seq_embeds"

class SeqStat:
    """Composition Stat including (0)key (1)type and (2)length"""
    def __init__(self, name, token_type, seq_len, *sub_seq_stats):
        self._name = name
        self._token_type = token_type
        self._seq_len = seq_len
        if self._token_type != SEQ_TYPE:
            assert sub_seq_stats is None
        self._sub_seq_stats = sub_seq_stats

    @property
    def name(self):
        return self._name

    @property
    def token_type(self):
        return self._token_type

    @property
    def seq_len(self):
        return self._seq_len

    @property
    def sub_seq_stats(self):
        return self._sub_seq_stats


def assert_type_valid(token_type):
    assert token_type == WORD_TYPE or token_type == ID_TYPE \
        or token_type == VALUE_INT_TYPE or token_type == VALUE_FLOAT_TYPE \
        or token_type == SEQ_IDS or token_type == SEQ_WORDS


def word2id_gen_f(vocabulary):
    def word2id_gen(gen_f):
        def gen():
            for word_token in gen_f():
                id_token = vocabulary.word2id_lookup(word_token)
                yield id_token 
        return gen
    return word2id_gen


def id2word_gen_f(vocabulary):
    def id2word_gen(gen_f):
        def gen():
            for id_token in gen_f():
                word_token = vocabulary.id2word_lookup(int(id_token))
                yield word_token 
        return gen
    return id2word_gen

