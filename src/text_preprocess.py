import re
from unicodedata import normalize
import pymorphy3
from nltk.stem import WordNetLemmatizer


class TextProcessor:
    def __init__(self, delete_digits: bool = True):
        self.delete_digits = delete_digits

        self.lemmatizer = WordNetLemmatizer()
        self.morph_ru = pymorphy3.MorphAnalyzer(lang='ru')

    def clean_text(self, text: str) -> str:
        """Очистка текста от лишней информации"""
        text = str(text)
        text = normalize("NFKC", text)

        text = re.sub(r'\s*\n', ' ', text)
        text = re.sub(r'\t', ' ', text)
        text = re.sub(r'\s*\r', ' ', text)
        text = text.replace("ё", "е").replace("Ё", "Е")
        text = re.sub(r"[^\w\s]", " ", text)
        if self.delete_digits:
            text = re.sub(r'\d+', ' ', text)
        text = re.sub("[^а-яА-Яa-zA-Z]", " ", text)
        text = re.sub(r' {2,}', ' ', text)
        text = text.strip()
        text = text.lower()
        return text

    def lemmatize_text(self, text: str) -> str:
        lemamtized_text = ' '.join(self.lemmatizer.lemmatize(word) for word in text.split())
        lemamtized_text = ' '.join(
            self.morph_ru.parse(word)[0].normal_form for word in lemamtized_text.split()
        )
        return lemamtized_text
