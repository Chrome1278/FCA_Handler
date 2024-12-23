import logging

import numpy as np
import pandas as pd
import scipy
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split

from src.fca_fit_predict import FitPredictFCA
from src.fca_impl import BinaryFCAClassifier
from src.text_preprocess import TextProcessor
from src.utils.metrics import binarize_target


logger = logging.getLogger()


class TrainerFCA:
    def __init__(self, df: pd.DataFrame, text_col_name: str, category_col_name: str,
                 mode: str, max_features: int, test_size: float):
        self.df = df
        self.text_col_name = text_col_name
        self.category_col_name = category_col_name
        self.mode = mode  # 'terms' or 'texts'
        if self.mode not in ['terms', 'texts', 'terms_and_texts']:
            logger.error(f'Incorrect value mode="{self.mode}"! Use only mode="terms" or mode="texts"')
            raise Exception(f'Incorrect value mode="{self.mode}"! Use only mode="terms" or mode="texts"')
        self.max_features = max_features
        self.test_size = test_size
        self.text_col_lemmas_name = None

        self.text_processor = TextProcessor()
        self.combined_stopwords = []
        self.get_stopwords()

    def preprocess_col(self):
        self.df[f'{self.text_col_name}'] = self.df[f'{self.text_col_name}'].apply(
            self.text_processor.clean_text
        )

    def get_lemmas_col(self):
        self.text_col_lemmas_name = f'{self.text_col_name}_lemmas'
        self.df[self.text_col_lemmas_name] = self.df[f'{self.text_col_name}'].apply(
            self.text_processor.lemmatize_text
        )

    def get_stopwords(self, special_stopwords: list[str] = None):
        if special_stopwords is None:
            special_stopwords = []
        russian_stopwords = stopwords.words('russian')
        english_stopwords = stopwords.words('english')
        special_stopwords = [w.lower() for w in special_stopwords]
        self.combined_stopwords = list(set(russian_stopwords +
                                           english_stopwords +
                                           special_stopwords))
        return self.combined_stopwords

    def __call__(self, clf: BinaryFCAClassifier):
        logger.info(f'Run trainer with data shape: {self.df.shape}')
        self.preprocess_col()
        if self.mode == 'texts':
            logger.info('Lemmatizing texts')
            self.get_lemmas_col()

        if self.mode == 'terms':
            self.df['joined_terms'] = self.df['terms'].apply(lambda x: '\t'.join(x))

        train_df, test_df = train_test_split(
            self.df, test_size=self.test_size, random_state=42, stratify=self.df[[self.category_col_name]]
        )

        corpus_train, corpus_test = None, None
        if self.mode == 'texts':
            corpus_train = train_df[self.text_col_lemmas_name].values
            corpus_test = test_df[self.text_col_lemmas_name].values
        elif self.mode == 'terms':
            corpus_train = train_df['joined_terms'].values
            corpus_test = test_df['joined_terms'].values

        logger.info('Fitting vectorizer')
        vectorizer_count = None
        if self.mode == 'texts':
            vectorizer_count = CountVectorizer(
                lowercase=False,
                min_df=3,
                max_features=self.max_features,
                stop_words=self.combined_stopwords,
                ngram_range=(1, 2)
            )
        elif self.mode == 'terms':
            vectorizer_count = CountVectorizer(
                token_pattern=r'[^\t]+',
                min_df=1,
                max_features=self.max_features
            )
        sparse_train = vectorizer_count.fit_transform(corpus_train)
        sparse_test = vectorizer_count.transform(corpus_test)

        vectorizer_tfidf = TfidfTransformer()
        X_train = vectorizer_tfidf.fit_transform(sparse_train)
        X_test = vectorizer_tfidf.transform(sparse_test)

        feature_names = vectorizer_count.get_feature_names_out()

        if self.mode == 'terms':
            terms_weight_dict = {}
            for terms, weights in zip(self.df['terms'], self.df['terms_weights']):
                for term, weight in zip(terms, weights):
                    terms_weight_dict[term.lower()] = int(weight)

            weight_vector = np.array([terms_weight_dict.get(term, 1) for term in feature_names])
            weight_matrix = scipy.sparse.diags(weight_vector)

            X_train = X_train.tocsr()
            X_test = X_test.tocsr()
            X_train = X_train.dot(weight_matrix)
            X_test = X_test.dot(weight_matrix)

            logger.info(f'Number of terms with weights: {len(terms_weight_dict)}')

        logger.info(f'Number of feature names: {len(feature_names)}')

        y_train, y_test, class_names, binarizer = binarize_target(
            train_df, test_df, self.category_col_name, return_binarizer=True
        )
        logger.info(f"Class names: {class_names}")
        logger.info('Starting FCA model')
        fca_model = FitPredictFCA(
            clf,
            vectorizer_count,
            X_train, X_test,
            y_train, y_test,
            class_names
        )
        fca_model.fit_fca()

        quality_df_final = fca_model.get_fca_model_metrics()

        return quality_df_final
