from nltk.corpus import stopwords
from math import log
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from nltk.stem.porter import *


def tokenize(s, remove_stopwords=True):
    if isinstance(s, str):
        s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s)
        s = s.lower()
        s = s.replace("  ", " ")
        s = s.replace(",", "")
        s = s.replace("$", " ")
        s = s.replace("?", " ")
        s = s.replace("-", " ")
        s = s.replace("//", "/")
        s = s.replace("..", ".")
        s = s.replace(" / ", " ")
        s = s.replace(" \\ ", " ")
        s = s.replace(".", " . ")
        s = re.sub(r"(^\.|/)", r"", s)
        s = re.sub(r"(\.|/)$", r"", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = s.replace(" x ", " xbi ")
        s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
        s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
        s = s.replace("*", " xbi ")
        s = s.replace(" by ", " xbi ")
        s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
        s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
        s = s.replace(" v ", " volts ")
        s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
        s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
        s = s.replace("  ", " ")
        s = s.replace(" . ", " ")
        strNum = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                  'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}
        s = " ".join([str(strNum[z]) if z in strNum else z for z in s.split(" ")])
        s = s.lower()
        words = s.split()
        # Optionally remove stop words (true by default)
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if w not in stops]
        # Return a list of words
        s = ' '.join(w for w in words)
        return s
    else:
        return 'Null'


def tokenize_origin(review, remove_stopwords=True):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #  Remove non-letters
    review_text = re.sub("[^a-zA-Z0-9]", " ", review)
    #  Convert words to lower case and split them
    words = review_text.lower().split()
    #  Optionally remove stop words (true by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if w not in stops]
    #  Return a list of words
    return words


def counter_appearance(list1, list2):
    count = 0
    for i in list1:
        if i in list2:
            count += 1

    return count


def counter_appear_times(list1, list2):
    count = 0
    for i in list1:
        for j in list2:
            if i == j:
                count += 1

    return count


def tfidf(df, column):

    docs_tf = {}
    idf = {}
    doc_length = {}
    vocab = set()

    count = 0
    total_length = 0
    c = 0

    for index, row in df.iterrows():
        dd = {}
        total_words = 0

        words = row[column].split()

        for word in words:
            vocab.add(word)
            dd.setdefault(word, 0)
            dd[word] += 1
            total_words += 1

        for k, v in dd.items():
            dd[k] = 1. * v / total_words

        docs_tf[row['product_uid']] = dd
        doc_length[row['product_uid']] = total_words

        count += 1
        c += 1

        total_length += total_words

        if c % 1000 == 0:
            print('processing ', c, 'th documents(tf)',)

    co = 0

    for w in list(vocab):
        docs_with_w = 0
        for path, doc_tf in docs_tf.items():
            if w in doc_tf:
                docs_with_w += 1
        idf[w] = log((len(docs_tf) - docs_with_w + 0.5)/(docs_with_w + 0.5))

        co += 1
        if co % 1000 == 0:
            print('processing ', co, 'th word(idf)',)

    ave_length = total_length/count

    return docs_tf, idf, doc_length, ave_length


def BM25_score(df, docs_tf, idf, doc_length, ave_length):

    score = 0
    query = df['search_term']
    prod_uid = df['product_uid']
    words = query.split()
    for word in words:
        if word in docs_tf[prod_uid]:
            score += idf[word]*((docs_tf[prod_uid][word] * 2.5)/(docs_tf[prod_uid][word] + 1.5 *
                                                           (1 - 0.75 + 0.75 * doc_length[prod_uid]/ave_length)))

    return score


def tf(doc_tf, search_term):
    tf = 0
    for word in search_term:
        if word in doc_tf:
            tf += doc_tf[word]

    return tf


def idf(idf, search_term):
    idf_score = 0
    for word in search_term:
        if word in idf:
            idf_score += idf[word]

    return idf_score


def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_


class cust_regression_vals(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, hd_searches):
        d_col_drops = ['id', 'relevance', 'search_term', 'product_title', 'product_description',
                       'attr', 'brand', 'attribute', 'attr_title']
        hd_searches = hd_searches.drop(d_col_drops, axis=1).values
        return hd_searches


class cust_txt_col(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key].apply(str)
