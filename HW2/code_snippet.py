import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


# If you do not know how to install pandas and scikit-learn packages, reach out to TAs immediately! 
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

    # Your code goes here:
    # D1.txt
    print("Document: D1.txt")
    print("How many distinct n-grams are there for D1.txt with 2-gram based on characters? Answer:", len(char_bigram('D1.txt')))
    print("How many distinct n-grams are there for D1.txt with 3-gram based on characters? Answer:", len(char_trigram('D1.txt')))
    print("How many distinct n-grams are there for D1.txt with 2-gram based on words? Answer:", len(word_bigram('D1.txt')))
    print('\n')
    # D2.txt
    print("Document: D2.txt")
    print("How many distinct n-grams are there for D2.txt with 2-gram? Answer:", len(char_bigram('D2.txt')))
    print("How many distinct n-grams are there for D2.txt with 3-gram based on characters? Answer:", len(char_trigram('D2.txt')))
    print("How many distinct n-grams are there for D2.txt with 2-gram based on words? Answer:", len(word_bigram('D2.txt')))
    print('\n')



    print("Jaccard Similarity with FIRST review and SECOND review for 2-gram based on word: "
          , jaccard_similarity(word_bigram('D1.txt'), word_bigram('D2.txt')))


n_grams_sklearn_incomplete()
