import hashlib
import math
import random
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def jaccard_similarity(A, B):
    numerator = len(list(set(A).intersection(set(B))))
    denominator = len(list(set(A).union(set(B))))
    return numerator / denominator


def char_bigram(filename):
    with open(filename) as doc1:
        text_list = []
        text = doc1.read()
        text_list.append(text)
        vectorizer1 = CountVectorizer(analyzer='char', ngram_range=(2, 2))
        X = vectorizer1.fit_transform(text_list)
        i = vectorizer1.get_feature_names_out()

    return i


def char_trigram(filename):
    with open(filename) as doc1:
        text_list = []
        text = doc1.read()
        text_list.append(text)
        vectorizer1 = CountVectorizer(analyzer='char', ngram_range=(3, 3))
        X = vectorizer1.fit_transform(text_list)
        i = vectorizer1.get_feature_names_out()

    return i


def word_bigram(filename):
    with open(filename) as doc1:
        text_list = []
        text = doc1.read()
        text_list.append(text)
        vectorizer1 = CountVectorizer(analyzer='word', ngram_range=(2, 2))
        X = vectorizer1.fit_transform(text_list)
        i = vectorizer1.get_feature_names_out()

    return i


def n_grams_sklearn_incomplete():
    # Let's read some data
    url = "https://raw.githubusercontent.com/koaning/icepickle/main/datasets/imdb_subset.csv"
    df = pd.read_csv(url)  # This is how you read a csv file to a pandas frame
    corpus = list(df['text'])
    corpus_small = corpus[:4]  # This is a list of 4 movie reviews

    '''
    Read documentation for https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
    Complete 1.A by using CountVectorizer, its methods, and adjusting certain parameters.
    '''

    t_list = [20, 60, 150, 300, 600]
    D1 = char_trigram('D1.txt')
    D2 = char_trigram('D2.txt')

    for t in t_list:
        numerator = 0
        m1 = {}
        m2 = {}
        form = []
        for i in range(1, t):
            form.append(str(random.randint(1, math.ceil(t * math.log(t)))))

        for a in D1:
            for j in range(1, t):
                m = hashlib.sha1()
                m.update(form[j - 1].encode('utf-8'))
                m.update(a.encode('utf-8'))
                value = m.hexdigest()
                if not j in m1 or value < m1.get(j):
                    m1[j] = value

        for b in D2:
            for j in range(1, t):
                m = hashlib.sha1()
                m.update(form[j - 1].encode('utf-8'))
                m.update(b.encode('utf-8'))
                value = m.hexdigest()
                if not j in m2 or value < m2.get(j):
                    m2[j] = value

        for i in range(1, t):
            if m1[i] == m2[i]:
                numerator += 1

        print("When t=", t)
        print('Approximate Jaccard Similarity between D1 and D2:', numerator / t)
        print('\n')


n_grams_sklearn_incomplete()
