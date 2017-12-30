import abc

class DocTransformer(object):
    seq_lens = []
    iter_keys = []

    def __len__(self):
        return len(seq_lens)

    @abc.abstractmethod
    def get_iters(doc):
        pass

    @property
    @abc.abstractmethod
    def token_types(self):
        pass

class IdentityTransform(DocTransformer):
    iter_keys = ["token"]
    seq_lens = [1]

    @property
    def token_types(self):
        return [self._token_type]

    def __init__(self, token_type):
        self._token_type = [token_type]

    def get_iters(doc):
        return iter(doc)


class Word2VecTransform(DocTransformer):
    iter_keys = ["center", "context"]
    seq_lens = [1, 1]

    @property
    def token_types(self):
        return [self._token_type, self._token_type]

    def __init__(window_size, max_num_examples, token_type):
        self._window_size = window_size
        self._max_num_examples = max_num_examples
        self._token_type = token_type
    
    def get_iters(doc):
        doc_gen = iter(doc)
        num_example = 0
        center_word = next(doc_gen)
        forward_context_words = []
        backward_context_words = []
        for _ in range(self._window_size):
            forward_context_words.append(next(doc_gen))
        while len(forward_context_words) != 0 and num_example < self._max_num_examples:
            for context_word in backward_context_words + forward_context_words:
                yield [center_word], [context_word]
            num_example += 1
            backward_context_words.append(center_word)
            center_word = forward_context_words.pop(0)
            try:
                next_word = next(doc_gen)
                forward_context_words.append(next_word)
            except StopIteration:
                pass
            if len(backward_context_words) > self._window_size:
                backward_context_words.pop(0)


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

    def get_iters(doc):
        doc_gen = iter(doc)
        data = np.array([token for token in doc_gen])
        data_len = data.shape[0]
        nb_batches = (data_len - 1)//(self._batch_size*self._seq_len)
        if self._nb_batches == 0:
            raise ValueError("Not enough data for even a single batch")
        rounded_data_len = self._nb_batches*self._batch_size*self._seq_len
        xdata = np.reshape(data[0:rounded_data_len],
                           [self._batch_size, self._nb_batches*self._seq_len])
        ydata = np.reshape(data[1:rounded_data_len+1], 
                           [self._batch_size, self._nb_batches*self._seq_len])

        for epoch_id in range(self._nb_epochs):
            for batch_id in range(self._nb_batches):
                x = xdata[:, batch_id*self._seq_len:(batch_id+1)*self._seq_len]
                y = ydata[:, batch_id*self._seq_len:(batch_id+1)*self._seq_len]
                x = np.roll(x, -epoch_id, axis=0)
                y = np.roll(y, -epoch_id, axis=0)
                for x_one_seq, y_one_seq in zip(x, y):
                    yield x_one_seq, y_one_seq

class DocLabelsTransform(DocTransformer):
    iter_keys = ["src", "label", "eod_flag"]

    @property
    def seq_lens(self):
        return [self._seq_len, 1, 1]

    @property
    def token_types(self):
        return [self._token_type, "id", "id"]

    def __init__(self, batch_size, seq_len, num_examples, token_type, *docs):
        self._docs = docs
        self._batch_size = batch_size
        self._seq_len = seq_len
        self._num_examples = num_examples
        self._token_type = token_type

    def get_iters(doc):
        docs = self._docs
        for doc_ in docs:
            assert doc.token_type == doc_.token_type
        start_doc_id = 0
        num_docs = len(docs)
        doc_seq_iters = [doc_.get_sequenced_iter(self._seq_len) for doc_ in docs]
        doc_label_dict = [{k: doc_.get_label(k) for k in label_keys} for doc_ in docs]
        curr_doc_seq_iters = doc_seq_iters[start_doc_id:start_doc_id+self._batch_size]
        curr_doc_label_dict = doc_label_dict[start_doc_id:start_doc_id+self._batch_size]
        
        next_seq, next_labels_dict = next(curr_doc_seq_iters[0]), curr_doc_label_dict[0]
        count = 1
        while count < self._num_examples:
            index = count % num_docs
            seq, labels_dict = next_seq, next_labels_dict
            try:
                next_seq, next_labels_dict = next(curr_doc_seq_iters[index]), curr_doc_label_dict[index]
                labels_dict["eod_flag"] = 0
            except StopIteration:
                labels_dict["eod_flag"] = 1
                start_doc_id += self._batch_size
                curr_doc_seq_iters = doc_seq_iters[start_doc_id:start_doc_id+self._batch_size]
                curr_doc_label_dict = doc_label_dict[start_doc_id:start_doc_id+self._batch_size]
                next_seq, next_labels_dict = next(doc_seq_iters[index]), doc_label_dict[index]
            yield seq, labels_dict
            count += 1


