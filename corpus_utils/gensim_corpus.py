from gensim import corpora, matutils


def create_bow_corpus(src_text_path, dictionary):
    for line in open(src_text_path):
        yield dictionary.doc2bow(line.split())


def save_bow_corpus(corpus, corpus_path):
    corpora.MmCorpus.serialize(corpus_path, corpus)


def create_save_bow_corpus(src_text_path, dictionary, corpus_path):
    corpus = create_bow_corpus(src_text_path, dictionary)
    corpora.MmCorpus.serialize(corpus_path, corpus)


def load_corpus(corpus_path):
    return corpora.MmCorpus(corpus_path)


def corpus2dense_mm(model, corpus, num_terms):
    return matutils.corpus2dense(model[corpus], num_terms=num_terms).T
