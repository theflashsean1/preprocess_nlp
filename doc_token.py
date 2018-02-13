WORD_TYPE = "word_type"  # str
ID_TYPE = "id_type"      # int
EMBED_TYPE = "embed_type"  # ndarray of np.float64
VALUE_INT_TYPE = "value_int_type"
VALUE_FLOAT_TYPE = "value_float_type"
SEQ_TYPE = "seq_type"


class SeqStat:
    """Composition Stat including (0)key (1)type and (2)length"""
    def __init__(self, name, token_type, seq_len, sub_seq_stat=None):
        self._name = name
        self._token_type = token_type
        self._seq_len = seq_len
        if self._token_type != SEQ_TYPE:
            assert sub_seq_stat is None
        self._sub_seq_stat = sub_seq_stat

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
    def sub_seq_stat(self):
        return self._sub_seq_stat


def assert_type_valid(token_type):
    assert token_type == WORD_TYPE or token_type == ID_TYPE \
        or token_type == VALUE_INT_TYPE or token_type == VALUE_FLOAT_TYPE \
        or token_Type == EMBED_TYPE or token_Type == SEQ_TYPE


def create_word2id_f_gen_f(vocab_reader, flag_tokens):
    word2id_lookup_f = vocab_reader.word2id_lookup
    def word2id_f_gen_f(gen_f):
        def gen():
            for word_token in gen_f():
                if word_token in flag_tokens:
                    yield word_token
                else:
                    id_token = word2id_lookup_f(word_token)
                    yield id_token 
        return gen
    return word2id_f_gen_f


def create_id2word_f_gen_f(vocab_reader, flag_tokens):
    id2word_lookup_f = vocab_reader.id2word_lookup
    def id2word_f_gen_f(gen_f):
        def gen():
            for id_token in gen_f():
                if id_token in flag_tokens:
                    yield id_token
                else:
                    word_token = id2word_lookup_f(int(id_token))
                    yield word_token 
        return gen
    return id2word_f_gen_f


def create_word2embed_f_gen_f(embed_reader, flag_tokens):
    word2embed_lookup_f = embed_reader.word2embed_lookup
    def word2embed_f_gen_f(gen_f):
        def gen():
            for word_token in gen_f():
                if word_token in flag_tokens:
                    yield word_token
                else:
                    embed_token = word2embed_lookup_f(word_token)
                    yield embed_token
        return gen
    return word2embed_f_gen_f


def create_id2embed_f_gen_f(embed_reader, flag_tokens):
    id2embed_lookup_f = embed_reader.id2embed_lookup
    def id2embed_f_gen_f(gen_f):
        def gen():
            for id_token in gen_f():
                if id_token in flag_tokens:
                    yield id_token
                else:
                    embed_token = id2embed_lookup_f(id_token)
                    yield embed_token
        return gen
    return id2embed_f_gen_f


class TokenTransformer:
    @property
    def num_left_tokens(self):
        return self._num_right_tokens

    @property
    def num_right_tokens(self):
        return self._num_left_tokens

    def __init__(self, token_transformer_f, num_left_tokens, num_right_tokens):
        self._num_left_tokens = num_left_tokens
        self._num_right_tokens = num_right_tokens
        self._token_transformer_f = token_transformer_f

    def __getitem__(self, tokens_tuple):
        left, center, right = tokens_tuple
        assert len(left) >= self._num_left_tokens
        assert len(right) >= self._num_right_tokens
        left = left[len(left)-self._num_left_tokens-1:]
        right = right[:self._num_right_tokens]
        return self._token_transformer_f(left, center, right)


    def is_applicable(self, left_len, right_len):
        left_valid = (left_len == self._num_left_tokens)
        right_valid = (right_len == self._num_right_tokens)
        return  left_valid and right_valid


def get_max_transformers_lens(token_transformers):
    left_len = 0
    right_len = 0
    for transformer in token_transformers:
        if transformer.num_left_tokens > left_len:
            left_len = transformer.num_left_tokens
        if transformer.num_right_tokens > right_len
            right_len = transformer.num_right_tokens
    return left_len, right_len


def shift_context_center_tokens(tokens_tuple, token_gen, left_max_len, right_max_len):
    left, center, right = tokens_tuple
    left.append(center)
    if len(left) > left_len:
        left.pop(0)
    center = right.pop(0) if len(right) > 0 else None
    try:
        right.append(next(token_gen))
    except StopIteration:
        pass
    return center


def apply_transforms(token_transformers, tokens_gen,
                     num_each_gen, applied_index, yield_index):
    pass
