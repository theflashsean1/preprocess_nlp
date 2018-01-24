import abc
import numpy as np
from preprocess_nlp.doc_token import WORD_TYPE, ID_TYPE, VALUE_INT_TYPE, VALUE_FLOAT_TYPE
from preprocess_nlp.file_utils.common import batched_items_iter, merged_round_iter
from preprocess_nlp.vocab_utils.common import PAD, PAD_ID
import pdb


class DocTransformer(object):
    seq_lens = []
    iter_keys = []

    def __len__(self):
        return len(self.seq_lens)

    @abc.abstractmethod
    def get_iters(self, *docs):
        pass

    @property
    @abc.abstractmethod
    def token_types(self):
        pass

    @abc.abstractmethod
    def estimate_max_size(self, *docs):
        pass

    @staticmethod
    def doc_token_types_check(*docs):
        for doc_ in docs:
            assert doc_.token_type == docs[0].token_type


class IdentityTransform(DocTransformer):
    iter_keys = ["token"]
    seq_lens = [1]

    @property
    def token_types(self):
        return [self._token_type]

    def __init__(self, token_type):
        self._token_type = [token_type]

    def get_iters(self, *docs):
        self.doc_token_types_check(*docs)
        for doc in docs:
            for token in doc:
                yield token

    def estimate_max_size(self, *docs):
        self.doc_token_types_check(*docs)
        len_sum = 0 
        for doc in docs:
            len_sum += len(doc)
        return len_sum


class Word2VecTransform(DocTransformer):
    iter_keys = ["center", "context"]
    seq_lens = [1, 1]

    @property
    def token_types(self):
        return [self._token_type, self._token_type]

    def __init__(self, window_size, token_type):
        self._window_size = window_size
        self._token_type = token_type
    
    def get_iters(self, *docs):
        def word2vec_gen(doc):
            doc_gen = iter(doc)
            num_example = 0
            center_word = next(doc_gen)
            forward_context_words = []
            backward_context_words = []
            for _ in range(self._window_size):
                forward_context_words.append(next(doc_gen))
            while len(forward_context_words) != 0:
                for context_word in backward_context_words + forward_context_words:
                    num_example += 1
                    yield center_word, context_word
                    if num_example >= self._max_num_examples:
                        return
                backward_context_words.append(center_word)
                center_word = forward_context_words.pop(0)
                try:
                    next_word = next(doc_gen)
                    forward_context_words.append(next_word)
                except StopIteration:
                    pass
                if len(backward_context_words) > self._window_size:
                    backward_context_words.pop(0)
        word2vec_iters = [word2vec_gen(doc) for doc in docs]
        for center, context in merged_round_iter(*word2vec_iters):
            yield center, context

    def estimate_max_size(self, *docs):
        len_sum = 0
        for doc in docs:
            len_sum += (2*self._window_size*len(doc) - 2*self._window_size)
        return len_sum


class Sca2wordTransform(DocTransformer):
    iter_keys = ["u_i_token", "w_i", "v_i_token",  "u_j_token", "w_j", "v_j_token"]
    seq_lens = [1, 1, 1, 1, 1, 1]

    @property
    def token_types(self):
        return [self._token_type, self._val_token_type, self._token_type, self._token_type, self._val_token_type, self._token_type]

    def __init__(self, each_num_example, token_type, val_token_type):
        self._token_type = token_type
        self._val_token_type = val_token_type
        self._each_num_example = each_num_example
    
    @staticmethod
    def is_num(token):
        return token.lstrip('-').replace('.', '', 1).isdigit()

    @classmethod
    def is_u_w_v(cls, u_w_v):
        u, w, v = u_w_v
        return cls.is_num(w) and (not cls.is_num(u)) and (not cls.is_num(v))

    def get_iters(self, *docs):
        def find_next_u_w_v(doc_iter):
            try:
                u_w_v = [next(doc_iter), next(doc_iter), next(doc_iter)]
            except StopIteration:
                return None
            while True:
                #if is_num(w):
                #    pdb.set_trace()
                if self.is_u_w_v(u_w_v):
                    return u_w_v
                try:
                    u_w_v.pop(0)
                    u_w_v.append(next(doc_iter))
                except StopIteration:
                    return None

        for doc in docs:
            doc_gen = iter(doc)
            u_w_v_i = find_next_u_w_v(doc_gen)
            if not u_w_v_i:
                raise ValueError("Not even a single example")
            comparisons = [find_next_u_w_v(doc_gen) for _ in range(self._each_num_example)]
            count = 0
            while count < self._max_num_examples:
                if comparisons[0] == None:
                    print("only found " + str(count) + " examples")
                    break
                for u_w_v_j in comparisons:
                    if not u_w_v_j:
                        break
                    u_i, w_i, v_i = u_w_v_i
                    u_j, w_j, v_j = u_w_v_j
                    count+=1
                    if self._val_token_type == VALUE_INT_TYPE:
                        yield u_i, int(float(w_i)), v_i, u_j, int(float(w_j)), v_j
                    elif self._val_token_type == VALUE_FLOAT_TYPE:
                        yield u_i, float(w_i), v_i, u_j, float(w_j), v_j
                    else:
                        raise ValueError("Unsupported Value token type")

                u_w_v_i = comparisons.pop(0)
                comparisons.append(find_next_u_w_v(doc_gen))


class Sca2ScapairTransformer(DocTransformer):
    iter_keys = ["w_i", "w_j"]
    seq_lens = [1, 1]

    @property
    def token_types(self):
        return [VALUE_INT_TYPE, VALUE_INT_TYPE]

    def __init__(self, each_num_example, max_num_examples):
        self._each_num_example = each_num_example
        self._max_num_examples = max_num_examples
        # self._val_token_type = val_token_type

    def get_iters(self, *docs):
        def find_w(doc_iter):
            try:
                w = next(doc_iter)
                return w
            except StopIteration:
                return None
        for doc in docs:
            count = 0 
            doc_gen = iter(doc)
            w_i = find_w(doc_gen)
            if not w_i:
                raise ValueError("Not even a single example")
            w_js = [find_w(doc_gen) for _ in range(self._each_num_example)]
            while count < self._max_num_examples:
                if w_js[0] == None:
                    print("only found " + str(count) + " examples")
                    break
                for w_j in w_js:
                    if not w_j:
                        break
                    count+=1
                    yield int(w_i), int(w_j)
                w_i = w_js.pop(0)
                w_js.append(find_w(doc_gen))
            

class RnnLangModelTransform(DocTransformer):
    iter_keys = ["src", "tgt"]

    @property
    def seq_lens(self):
        return [self._seq_len, self._seq_len]

    @property
    def token_types(self):
        return [self._token_type, self._token_type]

    def __init__(self, batch_size, seq_len, nb_epochs, token_type):
        self._token_type = token_type
        self._seq_len = seq_len
        self._batch_size = batch_size
        self._nb_epochs = nb_epochs

    def get_iters(self, *docs):
        for doc in docs:
            doc_gen = iter(doc)
            data = np.array([token for token in doc_gen])
            data_len = data.shape[0]
            nb_batches = (data_len - 1)//(self._batch_size*self._seq_len)
            if nb_batches == 0:
                raise ValueError("Not enough data for even a single batch")
            rounded_data_len = nb_batches*self._batch_size*self._seq_len
            xdata = np.reshape(data[0:rounded_data_len],
                               [self._batch_size, nb_batches*self._seq_len])
            ydata = np.reshape(data[1:rounded_data_len+1], 
                               [self._batch_size, nb_batches*self._seq_len])

            for epoch_id in range(self._nb_epochs):
                for batch_id in range(nb_batches):
                    x = xdata[:, batch_id*self._seq_len:(batch_id+1)*self._seq_len]
                    y = ydata[:, batch_id*self._seq_len:(batch_id+1)*self._seq_len]
                    x = np.roll(x, -epoch_id, axis=0)
                    y = np.roll(y, -epoch_id, axis=0)
                    for x_one_seq, y_one_seq in zip(x, y):
                        yield x_one_seq, y_one_seq

    def estimate_max_size(self, *docs):
        len_sum = 0
        for doc in docs:
            len_sum += (len(doc)//self._seq_len)
        return len_sum


class DocLabelsTransform(DocTransformer):
    iter_keys = ["src", "label", "eod_flag"]

    @property
    def seq_lens(self):
        return [self._seq_len, 1, 1]

    @property
    def token_types(self):
        return [self._token_type, ID_TYPE, ID_TYPE]

    def __init__(self, batch_size, seq_len, num_examples, token_type):
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._num_examples = num_examples
        self._token_type = token_type

    def get_iters(self, *docs):
        self.doc_token_types_check(*docs)
        for batched_docs in batched_items_iter(self._batch_size, *docs):
            seq_iters = [doc.get_sequenced_iter(self._seq_len) for doc in batched_docs]
            labels = [doc.get_label("label") for doc in docs]
            i = 0
            next_seq, next_label = next(seq_iters[i]), labels[i]
            while True:
                seq, label = next_seq, next_label
                i = (i+1) % self._batch_size
                try:
                    next_seq, next_label = next(seq_iters[i]), labels[i]
                    yield seq, label, 0
                except StopIteration:
                    yield seq, label, 1
                    break

    def estimate_max_size(self, *docs):
        self.doc_token_types_check(*docs)
        len_sum = 0
        for batched_docs in batched_items_iter(self._batch_size, *docs):
            len_sum += (min([len(doc) for doc in batched_docs]) / self.self._seq_len) * self._batch_size
        return len_sum


class DocLabelsPadTransform(DocTransformer):
    iter_keys = ["src", "label", "eod_flag"]

    @property
    def seq_lens(self):
        return [self._seq_len, 1, 1]

    @property
    def token_types(self):
        return [self._token_type, ID_TYPE, ID_TYPE]

    def __init__(self, batch_size, seq_len, token_type):
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._token_type = token_type
        self._pad_token = PAD if token_type == WORD_TYPE else PAD_ID

    def get_iters(self, *docs):
        self.doc_token_types_check(*docs)
        for batched_docs in batched_items_iter(self._batch_size, *docs):
            seq_iters = [doc.get_sequenced_iter(self._seq_len) for doc in batched_docs]
            labels = [doc.get_label("label") for doc in docs]
            eod_flags = [0 for _ in range(len(batched_docs))]
            i = 0
            next_seq, next_label = next(seq_iters[i]), labels[i]
            while True:
                seq, label = next_seq, next_label
                i = (i+1) % self._batch_size
                try:
                    next_seq, next_label = next(seq_iters[i]), labels[i]
                    yield seq, label, 0
                except StopIteration:
                    next_seq, next_label = self._pad_token, labels[i]
                    yield seq, label, 1  # TODO when to actually set this
                    eod_flags[i] = 1
                    all_eod = True
                    for flag in eod_flags:
                        if flag == 0:
                            all_eod = False
                    if all_eod:
                        break

    def get_better_iters(self, *docs):
        sorted_docs = sorted(docs, key=(lambda doc: len(doc)), reverse=True)
        return self.get_iters(*sorted_docs)

    def estimate_max_size(self, *docs):
        self.doc_token_types_check(*docs)
        len_sum = 0
        for batched_docs in batched_items_iter(self._batch_size, *docs):
            len_sum += (max([len(doc) for doc in batched_docs]) // self._seq_len) * self._batch_size
        return len_sum


class QuestionAnswerTransform(DocTransformer):
    iter_keys = [""]

