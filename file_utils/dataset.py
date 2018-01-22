import os
import preprocess_nlp.file_utils as fu


def ml_data_split(data_path, train_size, eval_size, test_size, res_dir=None):
    with open(data_path, "r") as data_f:
        def write_to_f(data_type, data_size):
            extended_signature = data_type + "_" + str(data_size)
            if not res_dir:
                new_path = fu.common.extend_path_basename(data_path, extended_signature)
            else:
                f_name = os.path.basename(data_path)
                if not os.path.exists(res_dir):
                    os.makedirs(res_dir)
                new_path = os.path.join(res_dir, fu.extend_file_basename(f_name))                
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
    

def file_line_split(docs_f_path, res_dir=None):
    if not res_dir:
        res_dir = os.path.dirname(docs_f_path)
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    with open(docs_f_path) as data_f:
        for i, line in enumerate(data_f):
            doc_path = os.path.join(res_dir, str(i) + ".txt")
            with open(doc_path, "w") as doc_f:
                doc_f.write(line)

