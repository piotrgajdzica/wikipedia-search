import time
import os
import numpy as np
from scipy.sparse import csc_matrix as sp
import scipy
from nltk.stem.snowball import SnowballStemmer
import pickle
import scipy.sparse.linalg


def get_raw_text(text):
    for char in '.[]!#/”*“=?:,();\n\"\'':
        text = text.replace(char, ' ')

    return text.lower()


def get_raw_articles(articles):
    for article in articles:
        article['raw'] = get_raw_text(article['text'])

    return articles


def get_articles(data_dir):
    articles = [None] * 1000000

    for file in os.listdir(data_dir):
        if file.endswith(".txt"):
            articles[int(file.split('.')[0])] = ({'text': open(os.path.join(data_dir, file),encoding='UTF-8').read(),'id': int(file.split('.')[0])})
    for i in range(len(articles)):
        if articles[i] is None:
            return articles[:i]

    return articles


def only_letters(text):
    for char in text:
        if char not in 'abcedfghijklmnopqrst\'uvwxyz':
            return False
    return True


def get_base_words(articles):
    base_words = set()
    all_words = set()
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    counter = {}

    stop_words = open("stop_words.txt", 'r').read().split('\n')

    for word in stop_words:
        counter[word] = 0

    for article in articles:
        for word in article['raw'].split():

            if word not in all_words:
                all_words.add(word)
                if only_letters(word) and word not in stop_words:
                    stemmed = stemmer.stem(word)

                    base_words.add(stemmed)
                    counter[word] = 1
            else:
                if only_letters(word):
                    counter[word] += 1

    for word, count in counter.items():
        if count < 10:
            stemmed = stemmer.stem(word)
            if stemmed in base_words:
                base_words.remove(stemmed)

    for word, count in counter.items():
        if count >= 10:
            stemmed = stemmer.stem(word)
            if stemmed not in base_words:
                base_words.add(stemmed)

    base_words = list(base_words)

    d = {}
    i = 0

    for word in base_words:
        d[word] = i
        i += 1

    return d


def get_matrix(articles, base_words):
    A = scipy.sparse.lil_matrix((len(articles), len(base_words)))

    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    all_words = set()
    stem_table = {}
    keys = base_words.keys()
    article_id = 0

    for article in articles:

        for word in article['raw'].split():
            if word not in all_words:

                if only_letters(word):

                    if word not in stem_table.keys():
                        stem_table[word] = stemmer.stem(word)
                    if stem_table[word] in keys:
                        A[article_id, base_words[stem_table[word]]] += 1

                else:
                    all_words.add(word)

        article_id += 1

    return sp(A).transpose()


def inverse_frequency(A):
    diag = []

    for row in A:
        if row.nnz > 0:
            diag.append(np.log2(A.shape[1] / row.nnz))

    offsets = [0]
    B = scipy.sparse.dia_matrix((diag, offsets), shape=(A.shape[0], A.shape[0]))
    A = B.dot(A)

    return A


def normalize(A):
    A = scipy.sparse.csr_matrix(A.transpose())

    for i in range(A.shape[0]):
        n = scipy.sparse.linalg.norm(A[i])
        A[i] = A[i] / n

    return A


def print_time(prefix, start):
    end = time.time()

    print(prefix + ": " + str(end - start) + "s")
    return end


def main():
    data_dir = "./data/articles/"
    articles = get_articles(data_dir)

    articles = get_raw_articles(articles)

    base_words = get_base_words(articles)

    A = get_matrix(articles, base_words)

    A = inverse_frequency(A)

    A = normalize(A)

    scipy.sparse.save_npz("./serialized/bag_of_words", A)

    with open('./serialized/base_words', 'wb') as fp:
        pickle.dump(base_words, fp)


main()
