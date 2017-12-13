import os 


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
    data_path = data_path.replace(filename, filename+"_"+extended_signature)
    return data_path


def aggregate_files(file_name, *file_paths):
    with open(file_name, "w") as f_write:
        with AggregatedReadOpen(*file_paths) as f_read:
            for line in f_read:
                f_write.write(line)
