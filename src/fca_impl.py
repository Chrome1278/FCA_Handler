import logging
from collections import defaultdict
import itertools
from typing import List
import numpy as np
import pandas as pd


logger = logging.getLogger()


def format_formula_as_str(formula):
    formula_clear = []
    for conj in formula:
        conj_clear = '   &&   '.join(conj)
        formula_clear.append(conj_clear)

    formula_str = '  ||   \n '.join(formula_clear)
    return formula_str


class BinaryFCAClassifier:

    def __init__(self,
                 max_formula_len: int = 20,
                 beta: float = 5,
                 wrecl: float = 2,
                 waddprec: float = 10,
                 waddrecl: float = 5,
                 base_freq: int = 3,
                 conj_num: int = 100,
                 base_prec: float = 0.2,
                 verbose=False) -> None:
        """ FCA (Formula Constructing Algorithm) classifier for binary classification

        Args:
            max_formula_len (int, optional): max length of formula. Defaults to 20.
            beta (float, optional): beta param to calculate F1-beta. Defaults to 1.
            wrecl (float, optional): wrecl param. Defaults to 2.
            waddprec (float, optional): waddprec param. Defaults to 10.
            waddrecl (float, optional): waddrecl param. Defaults to 5.
            base_freq (int, optional): min doc frequency. Defaults to 3.
            conj_num (int, optional): max terms to cosider. Defaults to 100.
            base_prec (float, optional): base precision threshold for first conjunction. Defaults to 0.2.
            verbose (bool, optional): verbosity param. Defaults to True.
        """
        self.max_formula_len = max_formula_len
        self.beta = beta
        self.wrecl = wrecl
        self.waddprec = waddprec
        self.waddrecl = waddrecl

        self.base_freq = base_freq
        self.conj_num = conj_num
        self.base_prec = base_prec

        self.verbose = verbose
        self.disable = not self.verbose
        self._feature_names = None

    def _create_inverse_idx(self, X):

        if self.verbose:
            logger.info('Creating inverse index')

        inverse_idx = dict()
        for term_idx in range(X.shape[1]):
            docs_with_term = set(X[:, term_idx].nonzero()[0])
            inverse_idx[term_idx] = docs_with_term

        return inverse_idx

    def _calc_f_for_one_word(self) -> pd.DataFrame:

        term_indices_in_topic = self.X[list(self.docs_with_topic), :].nonzero()[1]
        term_indices_in_topic = list(set(term_indices_in_topic))

        all_indices = set(range(self.n_docs))

        prec_recl_f = defaultdict(list)

        for term_idx in term_indices_in_topic:

            docs_with_term = self._inverse_idx[term_idx]
            docs_without_term = all_indices - docs_with_term

            a = len(docs_with_term.intersection(self.docs_with_topic))

            if a == 0:
                continue

            b = len(docs_with_term) - a
            c = len(docs_without_term.intersection(self.docs_with_topic))

            recl = a / (a + c)
            prec = a / (a + b)

            f_beta = 2 * prec * recl / (prec + recl)

            prec_recl_f['conj'].append((term_idx,))
            prec_recl_f['recl'].append(recl)
            prec_recl_f['prec'].append(prec)
            prec_recl_f['f_beta'].append(f_beta)
            prec_recl_f['docs_covered'].append(a)

        conj_one_word_df = pd.DataFrame(prec_recl_f)

        return conj_one_word_df

    def _find_conj_of_two_words(self, terms_to_conjunction) -> pd.DataFrame:

        prec_recl_f_conj = defaultdict(list)
        all_indices = set(range(self.n_docs))

        for term1, term2 in itertools.combinations(terms_to_conjunction, 2):
            docs_with_term1 = set(self._inverse_idx[term1])
            docs_with_term2 = set(self._inverse_idx[term2])
            docs_with_terms = docs_with_term1.intersection(docs_with_term2)

            docs_without_terms = all_indices - docs_with_terms

            a = len(docs_with_terms.intersection(self.docs_with_topic))

            if a == 0:
                continue

            b = len(docs_with_terms) - a
            c = len(docs_without_terms.intersection(self.docs_with_topic))

            recl = a / (a + c)
            prec = a / (a + b)

            f_beta = 2 * prec * recl / (prec + recl)
            recl, prec, f_beta

            prec_recl_f_conj['conj'].append((term1, term2))
            prec_recl_f_conj['recl'].append(recl)
            prec_recl_f_conj['prec'].append(prec)
            prec_recl_f_conj['f_beta'].append(f_beta)
            prec_recl_f_conj['docs_covered'].append(a)

        conj_two_words_df = pd.DataFrame(prec_recl_f_conj)
        return conj_two_words_df

    def _find_conj_of_three_word(self, terms_to_conjunction) -> pd.DataFrame:

        prec_recl_f_conj = defaultdict(list)
        all_indices = set(range(self.n_docs))

        for term1, term2, term3 in itertools.combinations(terms_to_conjunction, 3):
            docs_with_term1 = self._inverse_idx[term1]
            docs_with_term2 = self._inverse_idx[term2]
            docs_with_term3 = self._inverse_idx[term3]

            docs_with_terms = docs_with_term1.intersection(docs_with_term2).intersection(docs_with_term3)
            docs_without_terms = all_indices - docs_with_terms

            a = len(docs_with_terms.intersection(self.docs_with_topic))

            if a == 0:
                continue

            b = len(docs_with_terms) - a

            c = len(docs_without_terms.intersection(self.docs_with_topic))

            recl = a / (a + c)
            prec = a / (a + b)

            f_beta = 2 * prec * recl / (prec + recl)
            recl, prec, f_beta

            prec_recl_f_conj['conj'].append((term1, term2, term3))
            prec_recl_f_conj['recl'].append(recl)
            prec_recl_f_conj['prec'].append(prec)
            prec_recl_f_conj['f_beta'].append(f_beta)
            prec_recl_f_conj['docs_covered'].append(a)

        conj_three_words_df = pd.DataFrame(prec_recl_f_conj)
        return conj_three_words_df

    def _calc_f_for_first(self, conj_all_df) -> List:
        f_first_list = []
        for idx, row in conj_all_df.iterrows():
            prec = row['prec']
            recl = row['recl']
            f_first = 1 / (self.beta / prec + 1 / recl)
            f_first_list.append(f_first)
        return f_first_list

    def _find_first_conjunction(self):

        if self.verbose:
            logger.info('Calculating first conjunction')

        conj_one_word_df = self._calc_f_for_one_word()

        conj_one_word_df = conj_one_word_df.sort_values(by='f_beta', ascending=False)
        conj_one_word_df = conj_one_word_df.reset_index(drop=True)

        terms_to_conjunction = conj_one_word_df['conj'][:self.conj_num].values

        terms_to_conjunction = [x[0] for x in terms_to_conjunction]

        conj_two_words_df = self._find_conj_of_two_words(terms_to_conjunction)
        conj_three_words_df = self._find_conj_of_three_word(terms_to_conjunction)

        conj_all_df = pd.concat((conj_one_word_df, conj_two_words_df, conj_three_words_df), ignore_index=True)

        conj_all_df = conj_all_df[conj_all_df['prec'] >= self.base_prec]
        conj_all_df = conj_all_df[conj_all_df['docs_covered'] >= self.base_freq]
        conj_all_df.sort_values(by='f_beta', ascending=False)

        #
        f_first_list = self._calc_f_for_first(conj_all_df)
        conj_all_df['f_first'] = f_first_list

        conj_all_df = conj_all_df.sort_values(by='f_first', ascending=False)
        conj_all_df = conj_all_df.reset_index(drop=True)

        conj_first = conj_all_df.at[0, 'conj']
        return conj_first, conj_all_df

    def get_docs_by_formula(self, formula, inverse_idx) -> set:

        docs = set()

        for conjunction in formula:
            conj_len = len(conjunction)
            docs_conj = set()
            if conj_len == 1:
                term, = conjunction

                docs_conj = inverse_idx[term]
            elif conj_len == 2:
                term1, term2 = conjunction
                docs_with_term1 = inverse_idx[term1]
                docs_with_term2 = inverse_idx[term2]
                docs_conj = docs_with_term1.intersection(docs_with_term2)
            elif conj_len == 3:
                term1, term2, term3 = conjunction
                docs_with_term1 = inverse_idx[term1]
                docs_with_term2 = inverse_idx[term2]
                docs_with_term3 = inverse_idx[term3]
                docs_conj = docs_with_term1.intersection(docs_with_term2).intersection(docs_with_term3)
            docs = docs.union(docs_conj)
        return docs

    def _find_conjunction(self, current_formula, conj_list_df) -> pd.DataFrame:

        all_indices = set(range(self.n_docs))

        cntr = len(self.docs_with_topic)
        cntfr_docs = self.get_docs_by_formula(current_formula, inverse_idx=self._inverse_idx).intersection(
            self.docs_with_topic)

        cntfr = len(cntfr_docs)

        diff = cntr - cntfr
        cntfr_neg_docs = all_indices - cntfr_docs

        conj_candidates = defaultdict(list)

        weight_numenator = self.wrecl + self.waddprec + self.waddrecl

        conj_list = conj_list_df['conj'].values
        recl_list = conj_list_df['recl'].values
        prec_list = conj_list_df['prec'].values
        conj_candidates['F_conj'] = []

        for conj, recl, prec in zip(conj_list, recl_list, prec_list):
            docs_with_conj = self.get_docs_by_formula([conj], inverse_idx=self._inverse_idx)
            addf_docs = cntfr_neg_docs.intersection(docs_with_conj)
            addf = len(addf_docs)

            if addf == 0:
                continue

            addfr_docs = self.docs_with_topic.intersection(addf_docs)
            addfr = len(addfr_docs)

            if addfr == 0:
                continue

            addprec = addfr / addf
            addrecl = addfr / diff

            conj_candidates['addprec'].append(addprec)
            conj_candidates['addrecl'].append(addrecl)

            conj_candidates['prec'].append(prec)
            conj_candidates['recl'].append(recl)

            first = self.wrecl / recl
            second = self.waddprec / addprec
            third = self.waddrecl / addrecl

            F_conj = weight_numenator / (first + second + third)
            conj_candidates['F_conj'].append(F_conj)
            conj_candidates['conj'].append(conj)

        conj_candidates_df = pd.DataFrame(conj_candidates)
        conj_candidates_df.sort_values(by='F_conj', ascending=False)
        conj_candidates_df = conj_candidates_df.reset_index(drop=True)
        return conj_candidates_df

    def eval_prec_recl(self, formula, y, inverse_idx=None, calc_errors=False):

        if inverse_idx is None:
            inverse_idx = self._inverse_idx

        n_docs = y.shape[0]
        docs_with_topic = set(np.where(y == 1)[0])

        all_indices = set(range(n_docs))

        docs_with_formula = self.get_docs_by_formula(formula, inverse_idx=inverse_idx)

        a = len(docs_with_formula.intersection(docs_with_topic))

        docs_without_formula = all_indices - docs_with_formula

        c = len(docs_without_formula.intersection(docs_with_topic))

        if a == 0:
            prec = 0
            recl = 0
        else:
            b = len(docs_with_formula) - a
            recl = a / (a + c)
            prec = a / (a + b)

        result = dict()
        result['formula'] = formula
        result['prec'] = prec
        result['recl'] = recl

        docs_without_topic = all_indices - docs_with_topic

        if calc_errors:
            false_positive_docs = docs_with_formula.intersection(docs_without_topic)
            false_negative_docs = docs_without_formula.intersection(docs_with_topic)

            result['false_positive_docs'] = false_positive_docs
            result['false_negative_docs'] = false_negative_docs

        result['predictions'] = list(docs_with_formula)
        return result

    def _find_final_formula(self, conj_all_df, conj_first):

        if self.verbose:
            logger.info('Calculating final formula')

        conj_all_list_df = conj_all_df.copy()
        current_formula = [conj_first]

        best_f1 = 0
        best_formula_len = 1

        for i in range(1, self.max_formula_len):

            current_stats = self.eval_prec_recl(formula=current_formula, y=self.y)
            prec = current_stats['prec']
            recl = current_stats['recl']

            if prec == 0 or recl == 0:
                f1 = 0
            else:
                f1 = 2 * prec * recl / (prec + recl)

            if f1 > best_f1:
                best_f1 = f1
                best_formula_len = i

            if self.verbose:
                logger.info(f'Iter={i}, precision={prec}, recall={recl}, f1-score={f1}')
                current_formula_words = []
                for tuple_ids in current_formula:
                    words_tuple = tuple(self._feature_names[i] for i in tuple_ids if i < len(self._feature_names))
                    current_formula_words.append(words_tuple)
                # logger.info(current_formula_words)
                logger.info(f"Iter={i}, Current formula: \n{format_formula_as_str(current_formula_words)}")

            if current_stats['prec'] < 0.1 or current_stats['recl'] > 0.9:
                break

            conj_candidates_df = self._find_conjunction(current_formula, conj_all_list_df)

            if len(conj_candidates_df) == 0:
                break

            best_conj = conj_candidates_df.iloc[0, :]

            if best_conj['addrecl'] == 0 or \
                    best_conj['prec'] < 0.1 and best_conj['recl'] > 0.9 or \
                    best_conj['recl'] > 0.99:
                break

            if self.verbose:
                print(best_conj)

            conj_to_add = best_conj['conj']
            current_formula.append(conj_to_add)

            conj_candidates_df = conj_candidates_df.drop([0])
            conj_all_list_df = conj_candidates_df

        best_formula = current_formula[:best_formula_len]
        return best_formula

    def fit(self, X, y, inverse_idx=None):
        """
        Fit FCA model for binary classification

        Args:
            X (array-like, sparse matrix): Training data of shape (n_samples, n_features)
            y (array-like): Target values of shape (n_samples,). Values must be 0 and 1.

        Returns:
        self : object
            Fitted estimator
        """

        self.X = X

        if inverse_idx:
            self._inverse_idx = inverse_idx
        else:
            self._inverse_idx = self._create_inverse_idx(X)

        self.n_docs = y.shape[0]
        self.docs_with_topic = set(np.where(y == 1)[0])

        self.y = y

        conj_first, conj_all_df = self._find_first_conjunction()

        self.formula_ = self._find_final_formula(conj_all_df, conj_first)

        return self

    def predict(self, X):
        """
        Predict class (0/1)

        Args:
            X (array-like, sparse matrix): Training data of shape (n_samples, n_features)

        Returns:
            y (array-like): Predicted target values of shape (n_samples,).
        """

        inverse_idx = self._create_inverse_idx(X)
        docs = self.get_docs_by_formula(formula=self.formula_, inverse_idx=inverse_idx)

        prediction = np.zeros(X.shape[0])
        prediction[list(docs)] = 1
        return prediction

    def get_formula(self, feature_names=None):
        """
        Get constructed formula

        Args:
            feature_names (array-like, optional): feature names to use in formula. Defaults to None.

        Returns:
            List[Tuple[Any]]: constructed formula.
        """

        if feature_names is not None:
            return [tuple(feature_names[idx] for idx in conj) for conj in self.formula_]

        return self.formula_
