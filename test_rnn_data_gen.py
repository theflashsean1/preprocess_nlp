import tensorflow as tf
from corpus_utils import gensim_corpus
from file_utils import  common, dataset
from tokenize_utils import mimic


if __name__ == '__main__':
    """
    alive_src_path = "test_data/alive.csv"
    expire_src_path = "test_data/expire.csv"

    # Split dataset
    alive_train_path, alive_eval_path, alive_test_path = dataset.textline_file_split(alive_src_path, 1000, 200, 200)
    expire_train_path, expire_eval_path, expire_test_path = dataset.textline_file_split(expire_src_path, 1000, 200, 200)

    # Tokenize
    alive_train_path = mimic.notes_tokenize(alive_train_path)
    alive_eval_path = mimic.notes_tokenize(alive_eval_path)
    alive_test_path = mimic.notes_tokenize(alive_test_path)
    expire_train_path = mimic.notes_tokenize(expire_train_path)
    expire_eval_path = mimic.notes_tokenize(expire_eval_path)
    expire_test_path = mimic.notes_tokenize(expire_test_path)
    preprocessed_paths = [alive_train_path, alive_eval_path, alive_test_path,
                          expire_train_path, expire_eval_path, expire_test_path]
    """

    alive_train_path = "test_data/alive_train_1000.csv"
    alive_eval_path = "test_data/alive_eval_200.csv"
    alive_test_path = "test_data/alive_test_200.csv"
    expire_train_path = "test_data/expire_train_1000.csv"
    expire_eval_path = "test_data/expire_eval_200.csv"
    expire_test_path = "test_data/expire_test_200.csv"


def txt2minibatch_sequence(src_path, tgt_path, batch_size, seq_len, eos_toekns):
    with open(src_path) as f:
        with open(tgt_path, "a") as f_write:
            curr_tokens = []
            def write_seq(tokens):
                data_len = len(tokens)
                nb_batches = (data_len-1)//(batch_size*seq_len)
                assert nb_batches>0, "Not enough data, (even single batch)"
                for batch_id in range(nb_batches):
                    for i in range(batch_size):
                        seq = tokens[
                            (i*nb_batches*seq_len + seq_len*batch_id):(i*nb_batches*seq_len + seq_len*(batch_id+1))
                        ]
                        f_write.write(" ".join(seq)+"\n")

            for line in f:
                curr_tokens += line.strip().split()
                if line[-1] in eos_toekns:
                    write_seq(curr_tokens)
                    curr_tokens = []
            write_seq(curr_tokens)

#txt2minibatch_sequence("hamlet.txt", "hamlet_change.txt",
#        batch_size=3, seq_len=3, eos_toekns=[])


