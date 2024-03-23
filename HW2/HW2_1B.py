import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


def jaccard_similarity(A, B):
    numerator = len(list(set(A).intersection(set(B))))
    denominator = len(list(set(A).union(set(B))))
    return numerator / denominator


# If you do not know how to install pandas and scikit-learn packages, reach out to TAs immediately!
def char_bigram(corpus):
    corpus_list = [corpus]
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 2))
    X = vectorizer.fit_transform(corpus_list)
    i = vectorizer.get_feature_names_out()
    i = i.tolist()
    return i


def char_trigram(corpus):
    corpus_list = [corpus]
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3))
    X = vectorizer.fit_transform(corpus_list)
    i = vectorizer.get_feature_names_out()
    i = i.tolist()
    return i


def word_bigram(corpus):
    corpus_list = [corpus]
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(2, 2))
    X = vectorizer.fit_transform(corpus_list)
    i = vectorizer.get_feature_names_out()
    i = i.tolist()
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
    print("2-gram Based on Character")
    print("Jaccard Similarity with FIRST review and SECOND review for 2-gram based on character: "
          , jaccard_similarity(char_bigram(corpus_small[0]), char_bigram(corpus_small[1])))
    print("Jaccard Similarity with FIRST review and THIRD review for 2-gram based on character: "
          , jaccard_similarity(char_bigram(corpus_small[0]), char_bigram(corpus_small[2])))
    print("Jaccard Similarity with FIRST review and FORTH review for 2-gram based on character: "
          , jaccard_similarity(char_bigram(corpus_small[0]), char_bigram(corpus_small[3])))
    print("Jaccard Similarity with SECOND review and THIRD review for 2-gram based on character: "
          , jaccard_similarity(char_bigram(corpus_small[1]), char_bigram(corpus_small[2])))
    print("Jaccard Similarity with SECOND review and FORTH review for 2-gram based on character: "
          , jaccard_similarity(char_bigram(corpus_small[1]), char_bigram(corpus_small[3])))
    print("Jaccard Similarity with THIRD review and FORTH review for 2-gram based on character: "
          , jaccard_similarity(char_bigram(corpus_small[2]), char_bigram(corpus_small[3])))

    print('\n')

    # Second
    print("3-gram Based on Character")
    print("Jaccard Similarity with FIRST review and SECOND review for 3-gram based on character: "
          , jaccard_similarity(char_trigram(corpus_small[0]), char_trigram(corpus_small[1])))
    print("Jaccard Similarity with FIRST review and THIRD review for 3-gram based on character: "
          , jaccard_similarity(char_trigram(corpus_small[0]), char_trigram(corpus_small[2])))
    print("Jaccard Similarity with FIRST review and FORTH review for 3-gram based on character: "
          , jaccard_similarity(char_trigram(corpus_small[0]), char_trigram(corpus_small[3])))
    print("Jaccard Similarity with SECOND review and THIRD review for 3-gram based on character: "
          , jaccard_similarity(char_trigram(corpus_small[1]), char_trigram(corpus_small[2])))
    print("Jaccard Similarity with SECOND review and FORTH review for 3-gram based on character: "
          , jaccard_similarity(char_trigram(corpus_small[1]), char_trigram(corpus_small[3])))
    print("Jaccard Similarity with THIRD review and FORTH review for 3-gram based on character: "
          , jaccard_similarity(char_trigram(corpus_small[2]), char_trigram(corpus_small[3])))

    print('\n')

    # Third
    print("2-gram Based on Words")
    print("Jaccard Similarity with FIRST review and SECOND review for 2-gram based on word: "
          , jaccard_similarity(word_bigram(corpus_small[0]), word_bigram(corpus_small[1])))
    print("Jaccard Similarity with FIRST review and THIRD review for 2-gram based on word: "
          , jaccard_similarity(word_bigram(corpus_small[0]), word_bigram(corpus_small[2])))
    print("Jaccard Similarity with FIRST review and FORTH review for 2-gram based on word: "
          , jaccard_similarity(word_bigram(corpus_small[0]), word_bigram(corpus_small[3])))
    print("Jaccard Similarity with SECOND review and THIRD review for 2-gram based on word: "
          , jaccard_similarity(word_bigram(corpus_small[1]), word_bigram(corpus_small[2])))
    print("Jaccard Similarity with SECOND review and FORTH review for 2-gram based on word: "
          , jaccard_similarity(word_bigram(corpus_small[1]), word_bigram(corpus_small[3])))
    print("Jaccard Similarity with THIRD review and FORTH review for 2-gram based on word: "
          , jaccard_similarity(word_bigram(corpus_small[2]), word_bigram(corpus_small[3])))

    print('\n')


n_grams_sklearn_incomplete()
