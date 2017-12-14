import numpy as np
import vocab_utils

def words2batched_sequence_tfrecords():
    pass

def rnn_batch_sequencer_file():
    pass

def words2batched_seq(words, save_path, batch_size, seq_len):
    """
    Converts from raw document -> fixed batch size document where next batch is the continuation of previous batch
    ! For a txt file that has only one document in it.
    :param txt_path:
    :param save_path:
    :param batch_size:
    :param seq_len:
    :return:
    """
    pass


def doc2batched_seq_tfrecords(doc_path, save_path, batch_size, seq_len):
    pass


def txt2batch_sequence_space_efficient(txt_path, save_path, batch_size, seq_len):
    pass


def txt2batch_sequence2(txt_path, save_path, batch_size, seq_len, eod_tokens):
    """
    ! For a txt file that has multiple documents in it.
    """
    with open(txt_path) as f_read, open(save_path, "a") as f_write:
        def write_seq(tokens):
            data_len = len(tokens)
            nb_batches = (data_len-1)//(batch_size*seq_len)
            assert nb_batches > 0, "Not enough data, (even single batch)"
            for batch_id in range(nb_batches):
                for i in range(batch_size):
                    seq = tokens[
                          (i*nb_batches*seq_len + seq_len*batch_id):(i*nb_batches*seq_len + seq_len*(batch_id+1))
                          ]
                    f_write.write(" ".join(seq)+"\n")

        curr_tokens = []
        for line in f_read:
            curr_tokens += line.strip().split()
            if line[-1] in eod_tokens:
                write_seq(curr_tokens)
                curr_tokens = []
        write_seq(curr_tokens)


