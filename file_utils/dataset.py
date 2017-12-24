import os
from preprocess_nlp import file_utils


def textline_file_split(data_path, train_size, eval_size, test_size):
    with open(data_path, "r") as data_f:
        def write_to_f(data_type, data_size):
            extended_signature = data_type + "_" + str(data_size)
            new_path = file_utils.common.extend_path_basename(data_path, extended_signature)
            with open(new_path, "w") as f_w:
                for _ in range(data_size):
                    try:
                        f_w.write(next(data_f))
                    except:
                        raise Exception("Not enough data")
            return new_path
        train_file = write_to_f("train", train_size)
        eval_file = write_to_f("eval", eval_size)
        test_file = write_to_f("test", test_size)
    return train_file, eval_file, test_file
