import logging
from datetime import datetime


def setup_logging():
    log_filename = datetime.now().strftime(
        './data/logs/training_log_%Y-%m-%d-%H-%M.log'
    )
    logging.basicConfig(
        filename=log_filename,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s -- %(levelname)s: %(message)s",
        encoding='utf-8',
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s -- %(levelname)s: %(message)s"))
    logger = logging.getLogger()
    logger.addHandler(console_handler)


# if __name__ == "__main__":
#     setup_logging()
