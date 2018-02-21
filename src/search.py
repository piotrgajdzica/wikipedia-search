import heapq

import numpy as np
import time
import os
import constants
import scipy.sparse
import scipy.sparse.linalg
import pickle
from nltk.stem.snowball import SnowballStemmer


def get_bag_question(question, base_words):
    stemmer = SnowballStemmer("english", ignore_stopwords=True)

    bag = [0.0] * len(base_words)
    for word in question.split():
        stemmed = stemmer.stem(word)
        if stemmed in base_words.keys():
            bag[base_words[stemmed]] += 1

    return scipy.sparse.csc_matrix(bag)


def print_time(prefix, start):
    end = time.time()

    print(prefix + ": " + str(end - start) + "s")
    return end


def normalize(vector):
    return vector / scipy.sparse.linalg.norm(vector)


def get_matrix():
    return scipy.sparse.load_npz(os.path.join(constants.SERIALIZED_DIR, constants.MATRIX_FILE_NAME + ".npz"))


def get_base_words():
    return pickle.load(open(os.path.join(constants.SERIALIZED_DIR, constants.VECTOR_FILE_NAME), 'rb'))


def get_svd(k):
    u = pickle.load(open(os.path.join(constants.SOURCE_DIR + constants.SVD_DIR, str(k) + "_u"), 'rb'))
    S = pickle.load(open(os.path.join(constants.SOURCE_DIR + constants.SVD_DIR, str(k) + "_S"), 'rb'))
    v = pickle.load(open(os.path.join(constants.SOURCE_DIR + constants.SVD_DIR, str(k) + "_v"), 'rb'))
    return u, S, v


def search(question, k, A, base_words):

    A = get_matrix()

    base_words = get_base_words()

    bag_question = normalize(scipy.sparse.csr_matrix(get_bag_question(question, base_words).transpose()))
    matches = []

    nums = A.dot(bag_question)

    for i in nums.nonzero()[0]:
        if float(nums[i, 0]) > 0.001:
            matches.append({'id': i})

    ret = heapq.nlargest(k, matches, key=lambda x: nums[x['id'], 0])
    return ret


def search_svd(question, n, k):

    u, S, v = get_svd(k)
    base_words = get_base_words()

    bag_question = normalize(scipy.sparse.csr_matrix(get_bag_question(question, base_words)))

    matches = []
    h_array = []

    for i in range(S.shape[0]):
        h_array.append(S[i] * bag_question.dot(v[i]))

    h_array = np.array(h_array)
    u = u.transpose()

    for i in range(u.shape[0]):
        num = np.dot(u[i], h_array)
        matches.append({'cos': num, 'id': i})


    matches = sorted(matches, key=lambda x: x['cos'], reverse=True)


    return matches[0:n]


def prepare_results(results):
    ret = {'result': []}

    for result in results:
        filename = os.path.dirname(os.path.realpath("./data/articles/a")) + "\\" + str(result['id']) + ".txt"
        ret['result'].append(["\\" + str(result['id']), open(filename, encoding='UTF-8').readline()])
    return ret


def get_results(question, A, base_words):
    results = search(question, 10, A, base_words)

    return prepare_results(results)


def get_svd_results(question, k):
    results = search_svd(question, 10, k)

    return prepare_results(results)
