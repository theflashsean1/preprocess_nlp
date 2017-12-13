from gensim import corpora, models, matutils


def create_lda(corpus, dictionary, num_topics):
    lda = models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=num_topics,
                                   update_every=1, chunksize=10)
    return lda

def save_lda(lda_model, lda_path):
    lda_model.save(lda_path)


def load_lda(lda_path):
    lda = models.ldamodel.LdaModel.load(lda_path)
    return lda


def get_lda_topic_top_words(lda, dictionary, topic_id, terms=10):
    terms_freqs = lda.get_topic_terms(topic_id, terms)
    words = []
    for term_id, prop in terms_freqs:
        words.append(dictionary.get(term_id))
    return words

