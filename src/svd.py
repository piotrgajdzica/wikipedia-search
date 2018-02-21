import time
import os
import constants
import scipy.sparse
import scipy.sparse.linalg
from nltk.stem.snowball import SnowballStemmer
import pickle


def print_time(prefix, start):
    end = time.time()

    print(prefix + ": " + str(end - start) + "s")
    return end


def get_matrix():
    return scipy.sparse.load_npz(os.path.join(constants.SERIALIZED_DIR, constants.MATRIX_FILE_NAME + ".npz"))


def approximate(A, k):
    u, S, v = scipy.sparse.linalg.svds(A, k=k)
    u = u.transpose()

    file = constants.SOURCE_DIR + constants.SVD_DIR + str(k)

    f = open(file + "_u", 'wb')
    pickle.dump(u, f)
    f.close()
    f = open(file + "_S", 'wb')
    pickle.dump(S, f)
    f.close()
    f = open(file + "_v", 'wb')
    pickle.dump(v, f)
    f.close()


def get_bag_question(question, base_words):
    stemmer = SnowballStemmer("english", ignore_stopwords=True)

    bag = [0.0] * len(base_words)
    for word in question.split():
        stemmed = stemmer.stem(word)

        if stemmed in base_words.keys():
            bag[base_words[stemmed]] += 1

    return scipy.sparse.csc_matrix(bag)


def normalize(vector):
    print(vector.shape)
    return vector / scipy.sparse.linalg.norm(vector)


def get_base_words():
    return pickle.load(open(os.path.join(constants.SERIALIZED_DIR, constants.VECTOR_FILE_NAME), 'rb'))


def main():
    start = time.time()

    A = get_matrix()
    print(A.shape)
    start = print_time("matrix decompression ", start)

    for k in [10, 20, 40, 70, 100, 150, 200]:
        approximate(A, k)

    start = print_time("svd ", start)


main()
