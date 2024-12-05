import os
import xml.etree.ElementTree as ET
import re
import pandas as pd
from tqdm import tqdm
from typing import Optional, List, Dict


class NLDFileProcessor:
    """Класс для обработки NLD файлов и извлечения информации из них."""

    def __init__(self, folder_path: str,
                 selected_classes: Optional[List[str]] = None):
        """
        Инициализация процессора NLD файлов.

        Args:
            folder_path (str): Путь к папке с NLD файлами
            selected_classes (List[str], optional): Список выбранных классов ГРНТИ
        """
        self.folder_path = folder_path
        self.selected_classes = selected_classes

    @staticmethod
    def process_single_file(file_path: str) -> Dict:
        """
        Обработка отдельного NLD файла и извлечение основных полей.

        Args:
            file_path (str): Путь к NLD файлу

        Returns:
            Dict: Словарь с извлеченной информацией
        """
        tree = ET.parse(file_path)
        root = tree.getroot()
        file_info = {}
        for elem in root.iter():
            elem_text = elem.text
            if elem_text is not None and elem_text != '\n':
                file_info[elem.tag] = elem_text
        return file_info

    @staticmethod
    def fix_newline_after_words(text: str, word: str) -> str:
        """
        Исправление переносов строк после определенных слов.

        Args:
            text (str): Исходный текст
            word (str): Слово, после которого нужно исправить перенос

        Returns:
            str: Исправленный текст
        """
        pattern = re.escape(word) + r'\s*\n'
        return re.sub(pattern, word + ' ', text, count=1)

    @staticmethod
    def split_hdr_to_columns(hdr: str) -> Dict:
        """
        Разбивка строки заголовка на отдельные колонки.

        Args:
            hdr (str): Строка заголовка

        Returns:
            Dict: Словарь с разбитыми значениями
        """
        hdr = hdr.strip()
        hdr = NLDFileProcessor.fix_newline_after_words(hdr, 'BIBRECORD =')
        lines = hdr.strip().split('\n')
        result = {}

        for line in lines:
            if '=' in line:
                if "ГРНТИ=" in line:
                    start_index = line.find("ГРНТИ=")
                    grnti_value = line[start_index + 6:start_index + 12]
                    if grnti_value:
                        result['GRNTI'] = grnti_value
                key, value = line.split('=', 1)
                if not 'SMTA' in key and value.strip() is not None:
                    result[key.strip()] = value.strip()
        return result

    @staticmethod
    def extract_terms_data(row, objects: list[str], weight_threshold: int):
        """
        Разбивка строки заголовка на отдельные колонки.

        Args:
            row: Строка датафрейма
            objects (list[str]): список из типов терминов
            weight_threshold (int): пороговое значение веса терминов

        Returns:
            pd.Series(Dict): Словарь с двумя ключами, содержащими списки значений.
        """
        index_info = row['index'].split('\n')
        terms_info = []
        for obj in objects:
            obj_terms_info = [term for term in index_info if f"\t{obj}\t" in term]
            if obj_terms_info:
                terms_info.extend(obj_terms_info)
        # terms_info = [term for term in index_info if "\tTERM\t" in term]

        terms = [t.split('\t')[2] for t in terms_info]
        terms_weights = [t.split('\t')[-2] for t in terms_info]
        # terms_counts = [t.split('\t')[-1] for t in terms_info]

        filtered_terms = [term for i, term in enumerate(terms) if int(terms_weights[i]) >= weight_threshold]
        filtered_terms_weights = [weight for weight in terms_weights if int(weight) >= weight_threshold]

        return pd.Series({
            'terms': filtered_terms,
            'terms_weights': filtered_terms_weights,
            # 'terms_counts': terms_counts  # 1|2|1
        })

    def process_folder(self) -> pd.DataFrame:
        """
        Обработка всей папки с NLD файлами.

        Returns:
            pd.DataFrame: DataFrame с обработанными данными
        """
        all_nld_files = []
        for filename in tqdm(os.listdir(self.folder_path)):
            if filename.endswith('.nld'):
                file_path = os.path.join(self.folder_path, filename)
                nld_file_info = self.process_single_file(file_path)
                all_nld_files.append(nld_file_info)

        nld_files_df = pd.DataFrame(all_nld_files)
        splits = nld_files_df['hdr'].apply(self.split_hdr_to_columns)
        expanded_df = pd.json_normalize(splits)
        nld_files_df = pd.concat([nld_files_df, expanded_df], axis=1)
        nld_files_df = nld_files_df[['DocId', 'TITLE', 'TEXT_THEMAN_ANNO', 'GRNTI', 'index']]

        if self.selected_classes:
            nld_files_df = nld_files_df[nld_files_df.GRNTI.isin(self.selected_classes)]

        return nld_files_df
