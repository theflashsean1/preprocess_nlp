import abc
import numpy as np
import preprocess_nlp.doc_token as dt
# from preprocess_nlp.file_utils.common import batched_items_iter, merged_round_iter
import preprocess_nlp.file_utils.common as fc
import preprocess_nlp.vocab_utils as vu
import pdb


class DocTransformer(object):
    def __len__(self):
        return len(self.seq_stats)

    @property
    @abc.abstractmethod
    def seq_stats(self):
        pass

    @abc.abstractmethod
    def get_iters(self, *docs):
        pass

    @abc.abstractmethod
    def estimate_max_size(self, *docs):
        pass

    @staticmethod
    def doc_token_types_check(*docs):
        for doc_ in docs:
            assert doc_.token_type == docs[0].token_type


class IdentityTransform(DocTransformer):
    @property
    def seq_stats(self):
        if self._seq_stats is None:
            self._seq_stats = [dt.SeqStat("token", self._token_type, 1)]
        return self._seq_stats

    def __init__(self, token_type):
        self._seq_stats = None
        self._token_type = token_type

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
    @property
    def seq_stats(self):
        if not self._seq_stats:
            self._seq_stats = [
                dt.SeqStat("center", self._token_type, 1),
                dt.SeqStat("context", self._token_type, 1)
            ]
        return self._seq_stats

    def __init__(self, window_size, token_type):
        self._token_type = token_type
        self._window_size = window_size
        self._seq_stats = None
    
    def get_iters(self, *docs):
        def word2vec_gen(doc):
            doc_gen = iter(doc)
            num_example = 0
            # pdb.set_trace()
            center_word = next(doc_gen)
            forward_context_words = []
            backward_context_words = []
            for _ in range(self._window_size):
                forward_context_words.append(next(doc_gen))
            while len(forward_context_words) != 0:
                context_words = backward_context_words + forward_context_words
                for context_word in context_words:
                    num_example += 1
                    yield center_word, context_word
                backward_context_words.append(center_word)
                center_word = forward_context_words.pop(0)
                try:
                    next_word = next(doc_gen)
                    forward_context_words.append(next_word)
                except StopIteration:
                    pass
                if len(backward_context_words) > self._window_size:
                    backward_context_words.pop(0)
        # pdb.set_trace()
        word2vec_iters = [word2vec_gen(doc) for doc in docs]
        for center, context in merged_round_iter(*word2vec_iters):
            yield center, context

    def estimate_max_size(self, *docs):
        len_sum = 0
        for doc in docs:
            len_sum += (2*self._window_size*len(doc) - 2*self._window_size)
        return len_sum


class Sca2wordTransform(DocTransformer):
    iter_keys = ["u_i_token", "w_i", "v_i_token",
                 "u_j_token", "w_j", "v_j_token"]
    seq_lens = [1, 1, 1, 1, 1, 1]

    @property
    def token_types(self):
        return [self._out_token_type, self._out_val_token_type,
                self._out_token_type,
                self._out_token_type, self._out_val_token_type,
                self._out_token_type]

    def __init__(self, each_num_example, out_token_type, out_val_token_type,
                 vocab_reader=None):
        self._out_token_type = out_token_type
        self._out_val_token_type = out_val_token_type
        self._each_num_example = each_num_example
        if out_token_type == dt.ID_TYPE:
            assert vocab_reader is not None
        self._vocab_reader = vocab_reader

    @staticmethod
    def is_num(token):
        return token.lstrip('-').replace('.', '', 1).isdigit()

    @classmethod
    def is_u_w_v(cls, u_w_v):
        u, w, v = u_w_v
        return cls.is_num(w) and (not cls.is_num(u)) and (not cls.is_num(v))

    def get_iters(self, *docs):
        for doc in docs:
            assert doc.token_type == dt.WORD_TYPE
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

        def sca2word_gen(doc):
            token_wrap_f = lambda x: self._vocab_reader.word2id_lookup(x)\
                if self._out_token_type == dt.ID_TYPE else lambda x: x
            doc_gen = iter(doc)
            u_w_v_i = find_next_u_w_v(doc_gen)
            if not u_w_v_i:
                raise ValueError("Not even a single example")
            comparisons = [find_next_u_w_v(doc_gen) for _ in range(self._each_num_example)]
            count = 0
            while True:
                if comparisons[0] is None:
                    print("found " + str(count) + " examples")
                    break
                for u_w_v_j in comparisons:
                    if not u_w_v_j:
                        break
                    u_i, w_i, v_i = u_w_v_i
                    u_j, w_j, v_j = u_w_v_j
                    u_i, v_i = token_wrap_f(u_i), token_wrap_f(v_i)
                    u_j, v_j = token_wrap_f(u_j), token_wrap_f(v_j)
                    count += 1
                    if self._out_val_token_type == dt.VALUE_INT_TYPE:
                        yield u_i, int(float(w_i)), v_i, u_j, int(float(w_j)), v_j
                    elif self._out_val_token_type == dt.VALUE_FLOAT_TYPE:
                        yield u_i, float(w_i), v_i, u_j, float(w_j), v_j
                    else:
                        raise ValueError("Unsupported Value token type")

                u_w_v_i = comparisons.pop(0)
                comparisons.append(find_next_u_w_v(doc_gen))
        sca2word_iters = [sca2word_gen(doc) for doc in docs]
        for u_i, w_i, v_i, u_j, w_j, v_j in merged_round_iter(*sca2word_iters):
            yield u_i, w_i, v_i, u_j, w_j, v_j

    def estimate_max_size(self, *docs):
        len_sum = 0
        for _ in self.get_iters(*docs):
            len_sum += 1
        return len_sum


class Sca2ScapairTransformer(DocTransformer):
    iter_keys = ["w_i", "w_j"]
    seq_lens = [1, 1]

    @property
    def token_types(self):
        return [dt.VALUE_INT_TYPE, dt.VALUE_INT_TYPE]

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
        return [self._token_type, dt.ID_TYPE, dt.ID_TYPE]

    def __init__(self, batch_size, seq_len, token_type):
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._token_type = token_type

    def get_iters(self, *docs):
        self.doc_token_types_check(*docs)
        for batched_docs in batched_items_iter(self._batch_size, *docs):
            seq_iters = [doc.get_fixed_len_sequenced_iter(self._seq_len) for doc in batched_docs]
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
        return [self._token_type, dt.ID_TYPE, dt.ID_TYPE]

    def __init__(self, batch_size, seq_len, token_type):
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._token_type = token_type
        self._pad_token = vu.PAD if token_type == dt.WORD_TYPE else vu.PAD_ID

    def get_iters(self, *docs):
        self.doc_token_types_check(*docs)
        for batched_docs in batched_items_iter(self._batch_size, *docs):
            seq_iters = [doc.get_fixed_len_sequenced_iter(self._seq_len) for doc in batched_docs]
            labels = [doc.get_label("label") for doc in docs]
            for label in labels:
                assert label is not None
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
                    next_seq, next_label = [self._pad_token] * self._seq_len, labels[i]
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


class bAbITransform(DocTransformer):
    @property
    def seq_stats(self):
        if self._seq_stats is None:
            context_stat = dt.SeqStat(
                "context", self._out_token_type, self._c_max_len)
            self._seq_stats = (
                dt.SeqStat("question", self._out_token_type, self._q_max_len),
                dt.SeqStat("contexts", dt.SEQ_TYPE, self._cs_max_size, context_stat),
                dt.SeqStat("answer", self._out_token_type, self._a_max_len),
                dt.SeqStat("q_len", dt.VALUE_INT_TYPE, 1),
                dt.SeqStat("c_size", dt.VALUE_INT_TYPE, 1),
                dt.SeqStat("a_len", dt.VALUE_INT_TYPE, 1),
                dt.SeqStat("c_lens", dt.VALUE_INT_TYPE, self._c_max_size)
            )
        return self._seq_stats

    def __init__(self, q_max_len, cs_max_size, c_max_len, a_max_len,
                 out_token_type, nl_flag, vocab_reader=None):
        self._out_token_type = out_token_type
        if out_token_type == dt.ID_TYPE:
            assert vocab_reader is not None
            self._seq_type = dt.SEQ_EMBEDS
            self._pad_token = vu.PAD_ID
        else:
            self._seq_type = dt.SEQ_EMBEDS
            self._pad_token = vu.PAD
        self._nl_flag = nl_flag
        self._q_max_len = q_max_len
        self._c_max_len = c_max_len
        self._cs_max_size = cs_max_size
        self._a_max_len = a_max_len
        self._vocab_reader = vocab_reader

    def get_iters(self, *babi_docs):
        for doc in babi_docs:
            assert doc.token_type == dt.WORD_TYPE
        if self._out_token_type == dt.ID_TYPE:
            dtype = np.int64
            token_wrap_f = lambda x: self._vocab_reader.word2id_lookup(x)
            pad_token = vu.PAD_ID
        else:
            dtype = object
            token_wrap_f = lambda x: x
            pad_token = vu.PAD
        max_len = max(self._q_max_len, self._c_max_len)
        for babi_doc in babi_docs:
            context_seqs = np.full((self._cs_max_size, self._c_max_len),
                                   pad_token, dtype=dtype)
            c_index = 0
            for line_tokens in babi_doc.get_stop_token_sequenced_iter(self._nl_flag):
                seq_len, seq = 0, np.full(max_len, pad_token, dtype=dtype)
                seq_lens = np.full(self._cs_max_size, 0, dtype=np.int64)
                line_tokens = line_tokens[1:]
                yielded = False
                for i in range(len(line_tokens)):
                    if "\t" not in line_tokens[i]:
                        if i < max_len:
                            seq[i] = token_wrap_f(line_tokens[i])
                            seq_len += 1
                        continue
                    a_s_tokens = "".join(line_tokens[i:])
                    res = a_s_tokens.strip().split("\t")
                    if len(res) == 2:
                        ans_str, _ = res
                    elif len(res) == 3:
                        q_r_str, ans_str, _ = res
                        q_r_tokens = q_r_str.split()
                        for j in range(min(len(q_r_tokens), self._q_max_len-i)):
                            seq[i+j] = token_wrap_f(q_r_tokens[j])
                            seq_len += 1
                    else:
                        raise ValueError("Unexpected len split by tab")
                    ans = np.full(self._a_max_len, pad_token, dtype=dtype)
                    a_len, ans_tokens = 0, ans_str.split()
                    for j in range(min(len(ans_tokens), self._a_max_len)):
                        ans[j] = token_wrap_f(ans_tokens[j])
                        a_len += 1
                    yield seq, context_seqs, ans, seq_len, c_index+1, a_len
                    context_seqs = np.full(
                        (self._cs_max_size, self._c_max_len),
                        pad_token, dtype=dtype)
                    c_index = 0
                    yielded = True
                    break
                if yielded:
                    continue
                if c_index < self._cs_max_size:
                    context_seqs[c_index, :] = seq[:self._c_max_len]
                    seq_lens[c_index] = seq_len
                    c_index += 1


class bAbIEncodeEmbedsTransform(DocTransformer):
    """
    Require that input doc has token format dt.WORD_TYPE
    - because tab \t is string only, no associated ID in vocab
    output token type can be configured by token_type 
    """
    @property
    def seq_stats(self):
        if self._seq_stats is None:
            embed_stat = dt.SeqStat(
                "embed", dt.VALUE_FLOAT_TYPE, self._embed_reader.embed_size)
            self._seq_stats = (
                dt.SeqStat("question", dt.SEQ_TYPE, self._q_max_len, embed_stat),
                dt.SeqStat("contexts", dt.SEQ_TYPE, self._c_max_size, embed_stat),
                dt.SeqStat("answer", dt.SEQ_TYPE, self._a_max_len, embed_stat),
                dt.SeqStat("q_len", dt.VALUE_INT_TYPE, 1),
                dt.SeqStat("c_size", dt.VALUE_INT_TYPE, 1),
                dt.SeqStat("a_len", dt.VALUE_INT_TYPE, 1)
            )
        return self._seq_stats


    def __init__(self, q_max_len, cs_max_size, a_max_len,
                 out_token_type, nl_flag, embed_reader, encode_func):
        self._seq_stats = None
        self._out_token_type = out_token_type
        if out_token_type == dt.ID_TYPE:
            assert vocab_reader is not None
            self._seq_type = dt.SEQ_EMBEDS
            self._pad_token = vu.PAD_ID
        else:
            self._seq_type = dt.SEQ_EMBEDS
            self._pad_token = vu.PAD
        self._nl_flag = nl_flag
        self._q_max_len = q_max_len
        self._c_max_size = c_max_size
        self._a_max_len = a_max_len
        self._vocab_reader = vocab_reader
        self._babi_transformer = bAbITransform(q_max_len, cs)

    def get_iters(self, *babi_docs):
        pass


def resize_seq(seq, max_len, pad_token):
    if len(seq) < max_len:
        seq = seq + (max_len - len(seq))*[pad_token]
    elif len(seq) > max_len:
        seq = seq[:max_len]
    return seq


def resize_np_seq(seq, max_len, pad_token):
    if len(seq) < max_len:
        seq = np.pad(seq, (0, max_len-len(seq)),
                     mode="constant", constant_values=(pad_token, pad_token))
    elif len(seq) > max_len:
        seq = seq[:max_len]
    return seq
