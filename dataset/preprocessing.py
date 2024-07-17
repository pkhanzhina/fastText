import pandas as pd
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

stop_words = stopwords.words('english')
translator = str.maketrans('', '', string.punctuation)
lemmatizer = WordNetLemmatizer()


def clean_str(s):
    s = re.sub(r'[-\\]', ' ', s.lower())
    for i, word in enumerate(['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']):
        s = re.sub(r"{}".format(i), word + ' ', s)
    s = re.sub(r'[^a-z ]', '', s)
    return s.strip()


def tokenization(s, remove_stopwords):
    tokens = nltk.word_tokenize(s)
    if remove_stopwords:
        tokens = [t for t in tokens if t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens


def preprocessing(s, remove_stopwords=True):
    s = clean_str(s)
    tokens = tokenization(s, remove_stopwords)
    return tokens


if __name__ == '__main__':
    s = 'ffff\t'
    print()
    # path_to_data = 'data/AG News/train.csv'
    #
    # df = pd.read_csv(path_to_data)
    # df = df[:10]
    #
    # df['tokens'] = df['Description'].apply(preprocessing)
