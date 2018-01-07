WORD_TYPE = "word_type"  # str
ID_TYPE = "id_type"      # int
VALUE_INT_TYPE = "value_int_type"
VALUE_FLOAT_TYPE = "value_float_type"


def assert_type_valid(token_type):
    assert token_type == WORD_TYPE or token_type == ID_TYPE or token_type == VALUE_INT_TYPE or token_type == VALUE_FLOAT_TYPE


def word2id_gen_f(doc, vocabulary):
    doc_iter = iter(doc)
    def word2id_gen():
        for word_token in doc_iter:
            id_token = vocabulary.word2id_lookup(word_token)
            yield id_token 
    return word2id_gen


def id2word_gen_f(doc, vocabulary):
    doc_iter = iter(doc)
    def id2word_gen():
        for id_token in doc_iter:
            word_token = vocabulary.id2word_lookup(int(id_token))
            yield word_token 
    return id2word_gen

