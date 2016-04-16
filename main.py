import utils
import random
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from typo_correction import check_spell
from sklearn import pipeline, grid_search
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import make_scorer
from utils import fmean_squared_error, cust_regression_vals, cust_txt_col


if __name__ == '__main__':

    random.seed(42)

    RMSE = make_scorer(fmean_squared_error, greater_is_better=False)

    # load data
    df_train = pd.read_csv('data/train.csv', encoding="ISO-8859-1")
    num_train = df_train.shape[0]
    df_test = pd.read_csv('data/test.csv', encoding="ISO-8859-1")
    df_pro_desc = pd.read_csv('data/product_descriptions.csv')
    df_attr = pd.read_csv('data/attributes.csv')
    df_brand = df_attr[df_attr.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})
    data = pd.concat((df_train, df_test), axis=0, ignore_index=True)
    data = pd.merge(data, df_pro_desc, how='left', on='product_uid')
    data = pd.merge(data, df_brand, how='left', on='product_uid')

    # combine all attributes to a single description
    attr_all = df_attr[df_attr.name != 'MFG Brand Name'][['product_uid', 'value']].rename(columns={'value': 'attr'})
    attr_all['attr'] = attr_all['attr'].apply(lambda x: x if isinstance(x, str) else 'None')
    attr_all = attr_all.groupby('product_uid')['attr'].apply(lambda x: ' '.join(x))
    attr_all = pd.DataFrame(attr_all)
    attr_all = attr_all.reset_index()

    # combine all attributes' title to a single description
    attr_all_title = \
        df_attr[df_attr.name != 'MFG Brand Name'][['product_uid', 'name']].rename(columns={'name': 'attr_title'})
    attr_all_title['attr_title'] = attr_all_title['attr_title'].apply(lambda x: x if isinstance(x, str) else 'None')
    attr_all_title = attr_all_title.groupby('product_uid')['attr_title'].apply(lambda x: ' '.join(x))
    attr_all_title = pd.DataFrame(attr_all_title)
    attr_all_title = attr_all_title.reset_index()

    data = pd.merge(data, attr_all, how='left', on='product_uid')
    data = pd.merge(data, attr_all_title, how='left', on='product_uid')

    # tokenize all data
    data['search_term'] = data['search_term'].map(lambda x: check_spell(x))
    data['search_term'] = data['search_term'].map(lambda x: utils.tokenize(x))
    data['product_title'] = data['product_title'].map(lambda x: utils.tokenize(x))
    data['product_description'] = data['product_description'].map(lambda x: utils.tokenize(x))
    data['brand'] = data['brand'].map(lambda x: utils.tokenize(x))
    data['attr'] = data['attr'].map(lambda x: utils.tokenize(x))
    data['attr_title'] = data['attr_title'].map(lambda x: utils.tokenize(x))

    # add data length features
    data['len_of_query'] = data['search_term'].map(lambda x: len(x.split())).astype(np.int64)
    data['len_of_title'] = data['product_title'].map(lambda x: len(x.split())).astype(np.int64)
    data['len_of_description'] = data['product_description'].map(lambda x: len(x.split())).astype(np.int64)
    data['len_of_brand'] = data['brand'].map(lambda x: len(x.split())).astype(np.int64)
    data['len_of_attr'] = data['attr'].map(lambda x: len(x.split())).astype(np.int64)
    data['len_of_attr_title'] = data['attr_title'].map(lambda x: len(x.split())).astype(np.int64)

    # add word count features
    data['query_in_title'] = data.apply(lambda x: utils.counter_appearance(
        x['search_term'].split(), x['product_title'].split()), axis=1)
    data['query_in_description'] = data.apply(lambda x: utils.counter_appearance(
        x['search_term'].split(), x['product_description'].split()), axis=1)
    data['query_in_attr'] = data.apply(lambda x: utils.counter_appearance(
        x['search_term'].split(), x['attr'].split()), axis=1)
    data['query_in_attr_title'] = data.apply(lambda x: utils.counter_appearance(
        x['search_term'].split(), x['attr_title'].split()), axis=1)
    data['query_last_word_in_title'] = data.apply(lambda x: utils.counter_appearance(
        x['search_term'].split()[-1], x['product_title'].split()), axis=1)
    data['query_last_word_in_description'] = data.apply(lambda x: utils.counter_appearance(
        x['search_term'].split()[-1], x['product_description'].split()), axis=1)
    data['query_last_word_in_attr'] = data.apply(lambda x: utils.counter_appearance(
        x['search_term'].split()[-1], x['attr'].split()), axis=1)
    data['query_last_word_in_attr_title'] = data.apply(lambda x: utils.counter_appearance(
        x['search_term'].split()[-1], x['attr_title'].split()), axis=1)
    data['word_in_title'] = data.apply(lambda x: utils.counter_appear_times(
        x['search_term'].split(), x['product_title'].split()), axis=1)
    data['word_in_description'] = data.apply(lambda x: utils.counter_appear_times(
        x['search_term'].split(), x['product_description'].split()), axis=1)
    data['word_in_attr'] = data.apply(lambda x: utils.counter_appear_times(
        x['search_term'].split(), x['attr'].split()), axis=1)
    data['word_in_attr_title'] = data.apply(lambda x: utils.counter_appear_times(
        x['search_term'].split(), x['attr_title'].split()), axis=1)
    data['word_in_brand'] = data.apply(lambda x: utils.counter_appear_times(
        x['search_term'].split(), x['brand'].split()), axis=1)

    # add ratio features
    data['ratio_title'] = data['word_in_title']/data['len_of_query']
    data['ratio_description'] = data['word_in_description']/data['len_of_query']
    data['attribute'] = data['search_term']+"\t"+data['brand']
    data['ratio_brand'] = data['word_in_brand']/data['len_of_query']
    data['ratio_attr'] = data['word_in_attr']/data['len_of_query']
    data['ratio_attr_title'] = data['word_in_attr_title']/data['len_of_query']
    data['brand_ratio'] = data['word_in_brand']/data['len_of_brand']
    data['attr_ratio'] = data['word_in_attr']/data['len_of_attr']
    data['attr_title_ratio'] = data['word_in_attr_title']/data['len_of_attr_title']
    data['title_ratio'] = data['word_in_title']/data['len_of_title']
    data['description_ratio'] = data['word_in_description']/data['len_of_description']

    # add bm25 features
    desc_tf, desc_idf, desc_length, desc_ave_length = utils.tfidf(data, 'product_description')
    data['desc_BM25'] = data.apply(lambda x: utils.BM25_score(
        x, desc_tf, desc_idf, desc_length, desc_ave_length), axis=1)
    attr_tf, attr_idf, attr_length, attr_ave_length = utils.tfidf(data, 'attr')
    data['attr_BM25'] = data.apply(lambda x: utils.BM25_score(
        x, attr_tf, attr_idf, attr_length, attr_ave_length), axis=1)
    title_tf, title_idf, title_length, title_ave_length = utils.tfidf(data, 'product_title')
    data['title_BM25'] = data.apply(lambda x: utils.BM25_score(
        x, title_tf, title_idf, title_length, title_ave_length), axis=1)
    attr_title_tf, attr_title_idf, attr_title_length, attr_title_ave_length = utils.tfidf(data, 'attr_title')
    data['attr_title_BM25'] = data.apply(lambda x: utils.BM25_score(
        x, attr_title_tf, attr_title_idf, attr_title_length, attr_title_ave_length), axis=1)
    brand_tf, brand_idf, brand_length, brand_ave_length = utils.tfidf(data, 'brand')
    data['brand_BM25'] = data.apply(lambda x: utils.BM25_score(
        x, brand_tf, brand_idf, brand_length, brand_ave_length), axis=1)

    # add brand and search_term numerically as feature
    df_brand = pd.unique(data.brand.ravel())
    d = {}
    i = 1000
    for s in df_brand:
        d[s] = i
        i += 3
    data['brand_feature'] = data['brand'].map(lambda x: d[x])
    data['search_term_feature'] = data['search_term'].map(lambda x: len(x))

    # data.to_csv('data.csv')

    # data = pd.read_csv('data.csv', encoding="ISO-8859-1", index_col=0)
    df_train = data.iloc[:num_train]
    df_test = data.iloc[num_train:]
    id_test = df_test['id']
    y_train = df_train['relevance'].values
    X_train = df_train[:]
    X_test = df_test[:]

    # use random forest to train our model
    rfr = RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=42, verbose=1)
    # add tfidf term to our feature(by sk-learn build in function)
    tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
    tfidf_d2 = TfidfVectorizer(ngram_range=(2, 2), stop_words='english')
    tsvd = TruncatedSVD(n_components=10, random_state=42)
    clf = pipeline.Pipeline([
            ('union', FeatureUnion(
                        transformer_list=[
                            ('cst',  cust_regression_vals()),
                            ('txt1', pipeline.Pipeline([('s1', cust_txt_col(key='search_term')),
                                                        ('tfidf1', tfidf), ('tsvd1', tsvd)])),
                            ('txt2', pipeline.Pipeline([('s2', cust_txt_col(key='product_title')),
                                                        ('tfidf2', tfidf), ('tsvd2', tsvd)])),
                            ('txt3', pipeline.Pipeline([('s3', cust_txt_col(key='product_description')),
                                                        ('tfidf3', tfidf), ('tsvd3', tsvd)])),
                            ('txt4', pipeline.Pipeline([('s4', cust_txt_col(key='brand')), ('tfidf4', tfidf),
                                                        ('tsvd4', tsvd)])),
                            ],
                        transformer_weights={'cst': 1.0, 'txt1': 0.5, 'txt2': 0.25, 'txt3': 0.0, 'txt4': 0.5}
            )), ('rfr', rfr)])
    param_grid = {'rfr__max_features': [10], 'rfr__max_depth': [20]}
    model = grid_search.GridSearchCV(estimator=clf, param_grid=param_grid, n_jobs=-1, cv=5, verbose=20, scoring=RMSE)
    model.fit(X_train, y_train)

    prediction = model.predict(X_test)
    pd.DataFrame({"id": id_test, "relevance": prediction}).to_csv('submission.csv', index=False)
