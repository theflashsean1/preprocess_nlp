import os 

class MultiWriteOpen:
    def __init__(self, *files):
        self._files = [open(f, "w") for f in files]

    def __enter__(self):
        return self._files

    def __exit__(self, exc_type, exc_val, exc_tb):
        for f in self._files:
            f.close()


class AggregatedReadOpen:
    def __init__(self, *files):
        self._files = files

    def __iter__(self):
        for f in self._files:
            with open(f) as f_read:
                for line in f_read:
                    yield line
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class ReadReplaceOpen:
    def __init__(self, f_name):
        self._f_name = f_name
        self._temp_name = extend_path_basename(f_name, "temp")
        self._read_opened = open(self._f_name, "r")
        self._temp_opened = open(self._temp_name, "w")

    def __next__(self):
        return next(self._read_opened)

    def __iter__(self):
        return self

    def write(self, entry):
        self._temp_opened.write(entry)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._read_opened.close()
        self._temp_opened.close()
        os.rename(self._temp_name, self._f_name)


def extend_path_basename(data_path, extended_signature):
    basename = os.path.basename(data_path)
    filename, file_extension = os.path.splitext(basename)
    data_path = data_path.replace(basename, filename+"_"+extended_signature+file_extension)
    return data_path


def extend_file_basename(basename, extended_signature):
    filename, file_extension = os.path.splitext(basename)
    return filename + "_" + extended_signature + file_extension


def aggregate_files(file_name, *file_paths):
    with open(file_name, "w") as f_write:
        with AggregatedReadOpen(*file_paths) as f_read:
            for line in f_read:
                f_write.write(line)


def batched_items_iter(batch_size, *items):
    batched_items = []
    for item in items:
        batched_items.append(item)
        if len(batched_items) == batch_size:
            yield tuple(batched_items)
            batched_items = []


def merged_round_iter(*iters):
    while len(iters) != 0:
        next_iters = []
        for iterator in iters:
            try:
                res_tuple = next(iterator)
                next_iters.append(iterator)
                yield res_tuple
            except StopIteration:
                pass
        iters = next_iters


def limit_iter(iterator, max_num_examples):
    i = 0 
    for tuple_ in iterator:
        yield tuple_
        i += 1
        if i >= max_num_examples:
            break 
    if i < max_num_examples:
        print("Only found " + str(i) + " examples when expected " + str(max_num_examples))

