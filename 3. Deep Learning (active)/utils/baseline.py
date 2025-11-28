import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 

class PreprocessingText:
    def extract_label(self, text):
        # Находим метку в тексте 
        label = re.search(r'__label__\d', text).group()

        # Находим число в метке
        label = re.search(r'\d', label)
        if label:
            return int(label.group())
        else:
            return None
        
    def drop_label_in_text(self, text):
        # Удаляем метку из основного текста
        clean_text = re.sub(r'__label__\d', "", text)

        return clean_text

    def clean_text(self, text):
        # Приводим слова к нижнему регистру
        text = text.lower()
        
        # Удаляем все кроме английских букв
        text = re.sub(r'[^a-z0-9]', " ", text)

        # Очистка лишних пробелов
        text = re.sub(r'\s+', ' ', text).strip()

        return text