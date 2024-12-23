import logging
from ast import literal_eval
import pandas as pd

from src.fca_impl import BinaryFCAClassifier
from src.trainer import TrainerFCA
from src.utils.stopwords import special_stopwords
from src.utils.logger import setup_logging


DOCS_DF_PATH = 'data/docs/actual_docs_df_with_all_terms.csv'
TRAINER_MODE = 'terms'  # 'texts' or 'terms'
MAX_FEATURES_TF_IDF = 30000
TEST_SIZE = 0.2

setup_logging()
logger = logging.getLogger()

if __name__ == "__main__":
    logger.info(f'Reading data from path: {DOCS_DF_PATH}')
    docs_df = pd.read_csv(
        DOCS_DF_PATH,
        converters={'terms': literal_eval, 'terms_weights': literal_eval}
    )
    # docs_df = docs_df[::10]  # Взять лишь каждый 10 документ из данных для ускоренного демо запуска

    logger.info('Starting FCA trainer')
    trainer_fca = TrainerFCA(
        df=docs_df,
        text_col_name='TEXT_THEMAN_ANNO',
        category_col_name='GRNTI',
        mode=TRAINER_MODE,
        max_features=MAX_FEATURES_TF_IDF,
        test_size=TEST_SIZE
    )
    trainer_fca.get_stopwords(special_stopwords=special_stopwords)
    logger.info(f"TRAINER_MODE: {TRAINER_MODE}, MAX_FEATURES_TF_IDF: {MAX_FEATURES_TF_IDF}, TEST_SIZE: {TEST_SIZE}")

    clf = BinaryFCAClassifier(
        max_formula_len=30,
        beta=30,
        wrecl=2,
        waddprec=10,
        waddrecl=5,
        base_freq=3,
        conj_num=100,
        base_prec=0.2,
        verbose=True
    )
    logger.info(f"Params of FCA classifier: {clf.__dict__}")

    try:
        fca_metrics_df = trainer_fca(clf=clf)
    except Exception as e:
        logger.error(e)
        raise e
