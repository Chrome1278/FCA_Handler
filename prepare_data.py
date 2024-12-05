import glob

from src.nld_processor import NLDFileProcessor

# абсолютный путь до директории с nld файлами
FOLDER_PATH = 'C:/Users/Александр/Desktop/Учеба/1 курс/Научка/20240305/all_nld'
# ГРНТИ-коды классов, которые будут отобраны.
TARGET_CLASSES = ['340000', '550000', '290000', '270000', '310000', '500000']
# Виды отбираемых терминов
INDEX_OBJECTS = ['TERM', 'TERMTREE', 'TERMTREELOW']
# Пороговое значение веса для терминов
WEIGHT_THRESHOLD = 5
# путь сохранения результата с названием файла без расширения
SAVE_RESULT_PATH = './data/actual_docs_df_with_all_terms'


if __name__ == "__main__":
    processor = NLDFileProcessor(
        folder_path=FOLDER_PATH,
        selected_classes=TARGET_CLASSES
    )

    print('Count of NLD files in folder:', len(glob.glob(FOLDER_PATH + f'/*.nld', recursive=True)))
    actual_docs_df = processor.process_folder()

    actual_docs_df[['terms', 'terms_weights']] = actual_docs_df.apply(
        lambda row: processor.extract_terms_data(row, INDEX_OBJECTS, WEIGHT_THRESHOLD), axis=1
    )
    actual_docs_df = actual_docs_df[['DocId', 'TEXT_THEMAN_ANNO', 'GRNTI', 'terms']]
    print('Total count of documents in dataframe:', actual_docs_df.shape[0])

    actual_docs_df.to_csv(SAVE_RESULT_PATH + ".csv", index=False, encoding='utf-8-sig')
    actual_docs_df.to_excel(SAVE_RESULT_PATH + ".xlsx", index=False)
    print(f'Results was successfully saved to the path: {FOLDER_PATH}')
