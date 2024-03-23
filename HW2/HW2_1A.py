import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


# If you do not know how to install pandas and scikit-learn packages, reach out to TAs immediately!
def char_bigram(corpus):
    corpus_list = [corpus]
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 2))
    X = vectorizer.fit_transform(corpus_list)
    i = vectorizer.get_feature_names_out()

    return i


def char_trigram(corpus):
    corpus_list = [corpus]
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3))
    X = vectorizer.fit_transform(corpus_list)
    i = vectorizer.get_feature_names_out()

    return i


def word_bigram(corpus):
    corpus_list = [corpus]
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2))
    X = vectorizer.fit_transform(corpus_list)
    i = vectorizer.get_feature_names_out()

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
    # First
    print("First Movie Review")
    print("How many distinct n-grams are there for first movie review with 2-gram based on characters? Answer:", len(char_bigram(corpus_small[0])))
    print("How many distinct n-grams are there for first with 3-gram based on characters? Answer:", len(char_trigram(corpus_small[0])))
    print("How many distinct n-grams are there for first with 2-gram based on words? Answer:", len(word_bigram(corpus_small[0])))
    print('\n')
    # Second
    print("Second Movie Review")
    print("How many distinct n-grams are there for second with 2-gram? Answer:", len(char_bigram(corpus_small[1])))
    print("How many distinct n-grams are there for second with 3-gram based on characters? Answer:", len(char_trigram(corpus_small[1])))
    print("How many distinct n-grams are there for second with 2-gram based on words? Answer:", len(word_bigram(corpus_small[1])))
    print('\n')
    # Third
    print("Third Movie Review")
    print("How many distinct n-grams are there for third with 2-gram? Answer:", len(char_bigram(corpus_small[2])))
    print("How many distinct n-grams are there for third with 3-gram based on characters? Answer:", len(char_trigram(corpus_small[2])))
    print("How many distinct n-grams are there for third with 2-gram based on words? Answer:", len(word_bigram(corpus_small[2])))
    print('\n')
    # Forth
    print("Forth Movie Review")
    print("How many distinct n-grams are there for forth with 2-gram? Answer:", len(char_bigram(corpus_small[3])))
    print("How many distinct n-grams are there for forth with 3-gram based on characters? Answer:", len(char_trigram(corpus_small[3])))
    print("How many distinct n-grams are there for forth with 2-gram based on words? Answer:", len(word_bigram(corpus_small[3])))
    print('\n')


n_grams_sklearn_incomplete()
