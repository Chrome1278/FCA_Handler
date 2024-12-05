import logging
from datetime import datetime

import pandas as pd
import tqdm

from src.fca_impl import BinaryFCAClassifier, format_formula_as_str
from src.utils.metrics import eval_metrics


logger = logging.getLogger()


class FitPredictFCA:
    def __init__(self, clf: BinaryFCAClassifier,
                 vectorizer_count, x_train, x_test,
                 y_train, y_test, class_names):
        self.class_names = class_names
        self.y_test = y_test
        self.y_train = y_train
        self.X_test = x_test
        self.vectorizer_count = vectorizer_count
        self.clf = clf
        self.X_train = x_train

        self.topic_to_formula_raw = dict()
        self.topic_to_formula = dict()

    def fit_fca(self):
        inverse_idx = None
        feature_names = self.vectorizer_count.get_feature_names_out()
        self.clf._feature_names = feature_names
        for class_idx, class_name in tqdm.tqdm(enumerate(self.class_names)):
            self.clf.fit(self.X_train, self.y_train[:, class_idx], inverse_idx=inverse_idx)
            logger.info(f'Fit for class={class_name} is ready')
            formula_raw = self.clf.get_formula()
            formula = self.clf.get_formula(feature_names=feature_names)

            self.topic_to_formula_raw[class_name] = formula_raw
            self.topic_to_formula[class_name] = formula

            y_pred_train = self.clf.predict(self.X_train)
            f1_train, precision_train, recall_train = eval_metrics(self.y_train[:, class_idx], y_pred_train)
            logger.info(f'train predict for {class_name} is ready')
            logger.info(f'f1-score={f1_train}, precision={precision_train}, recall={recall_train}')

            y_pred_test = self.clf.predict(self.X_test)
            f1_test, precision_test, recall_test = eval_metrics(self.y_test[:, class_idx], y_pred_test)
            logger.info(f'test predict for {class_name} is ready')
            logger.info(f'f1-score={f1_test}, precision={precision_test}, recall={recall_test}')

    def get_fca_model_metrics(self) -> pd.DataFrame:
        inverse_idx_test = self.clf._create_inverse_idx(self.X_test)
        all_res = []
        for current_topic_idx, current_topic in enumerate(self.class_names):
            current_formula = self.topic_to_formula[current_topic]
            current_formula_raw = self.topic_to_formula_raw[current_topic]

            truncation_stats = []
            for idx in range(1, len(current_formula) + 1):
                res_tr = self.clf.eval_prec_recl(current_formula_raw[:idx], y=self.y_train[:, current_topic_idx],
                                                 inverse_idx=self.clf._inverse_idx)
                prec_tr = res_tr['prec']
                recl_tr = res_tr['recl']
                f1_tr = 2 * prec_tr * recl_tr / (prec_tr + recl_tr)

                res_tt = self.clf.eval_prec_recl(current_formula_raw[:idx], y=self.y_test[:, current_topic_idx],
                                                 inverse_idx=inverse_idx_test)
                prec_tt = res_tt['prec']
                recl_tt = res_tt['recl']
                f1_tt = 2 * prec_tt * recl_tt / (prec_tt + recl_tt)

                truncation_stats.append({'idx': idx,
                                         'prec_train': prec_tr,
                                         'recl_train': recl_tr,
                                         'f1_train': f1_tr,
                                         'prec_test': prec_tt,
                                         'recl_test': recl_tt,
                                         'f1_test': f1_tt,
                                         'current_formula': current_formula[:idx],
                                         })
            truncation_df = pd.DataFrame(truncation_stats)

            # select best by f1_train
            best_idx = truncation_df.loc[truncation_df['f1_train'].argmax()]['idx'].astype(int)
            truncated_formula = current_formula[:best_idx]
            truncated_formula_raw = current_formula_raw[:best_idx]

            # calc metrics again for best on TRAIN
            eval_train = self.clf.eval_prec_recl(formula=truncated_formula_raw, y=self.y_train[:, current_topic_idx],
                                                 inverse_idx=self.clf._inverse_idx)
            prec_train = eval_train['prec']
            recl_train = eval_train['recl']
            f1_train = 2 * prec_train * recl_train / (prec_train + recl_train)

            eval_test = self.clf.eval_prec_recl(formula=truncated_formula_raw, y=self.y_test[:, current_topic_idx],
                                                inverse_idx=inverse_idx_test)
            prec_test = eval_test['prec']
            recl_test = eval_test['recl']
            f1_test = 2 * prec_test * recl_test / (prec_test + recl_test)

            all_res.append({'topic': current_topic,
                            'formula': truncated_formula,
                            'str_formula': format_formula_as_str(truncated_formula),
                            'f1_train': f1_train,
                            'f1_test': f1_test,
                            'prec_train': prec_train,
                            'recl_train': recl_train,
                            'prec_test': prec_test,
                            'recl_test': recl_test,
                            'formula_len': len(truncated_formula),
                            'truncation_stats': truncation_df.to_dict()}
                           )

        quality_df = pd.DataFrame(all_res)
        quality_df_final = pd.concat(
            [quality_df,
             pd.DataFrame({'topic': ['f1-macro'],
                           'formula': ['-'],
                           'f1_train': [quality_df['f1_train'].mean()],
                           'f1_test': [quality_df['f1_test'].mean()],
                           'prec_train': ['-'],
                           'recl_train': ['-'],
                           'prec_test': ['-'],
                           'recl_test': ['-'],
                           'formula_len': ['-'],
                           'truncation_stats': ['-']})], ignore_index=True)
        saved_filename = datetime.now().strftime(
            f'fca_results_with_train_size_{self.X_train.shape[0]}_date_%Y-%m-%d-%H-%M.xlsx'
        )
        quality_df_final.to_excel(f"./data/metrics/{saved_filename}", index=False)
        logger.info(f'FCA model results was successfully saved to the path: ./data/metrics/{saved_filename}')
        return quality_df_final
