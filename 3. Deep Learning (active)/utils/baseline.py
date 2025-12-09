import pandas as pd
import numpy as np

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 

from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
#from sklearn.feature_selection import 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from statsmodels.stats.outliers_influence import variance_inflation_factor


import spacy # type: ignore
from multiprocessing import Pool

import logging

logging.basicConfig(level=logging.INFO, filename="./data/logging.log",filemode="w",
                    format="%(asctime)s %(levelname)s %(message)s", encoding='utf-8')

nlp = spacy.load("en_core_web_sm")

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
        
        # Удаляем все кроме английских букв и чисел
        text = re.sub(r'[^a-z0-9\s]', "", text)

        # Убираем лишние пробелы
        text = re.sub(r'\s+', ' ', text).strip()

        # Лемматизация текста
        doc = nlp(text)
        text = " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

        return text

    def process_texts_in_parallel(self, texts):
        # Используем nlp.pipe() для пакетной обработки текста
        with Pool() as pool:
            result = pool.map(self.clean_text, texts)
        return result
    

class PipelineManager():
    def __init__(self, values, vectozer, model):
        self.X_train = values[0]
        self.y_train = values[1]
        self.X_val = values[2]
        self.y_val = values[3]
        self.vectozer = vectozer
        self.model = model

    def category_columns(self):
        categorical_columns = self.X.select_dtypes(include=['object']).columns.to_list()
        numeric_columns = self.X.select_dtypes(include=['int', 'float']).columns.to_list()

        return categorical_columns, numeric_columns
    
    def create_pipeline(self):
        
        # Создание пайплайна
        pipe = Pipeline([
            ('vectorizer', self.vectozer),
            ('clf', self.model)
        ]) 

        # Обучение модели
        pipe.fit(self.X_train, self.y_train)

        # Предсказание и получение результатов метрики
        pred = pipe.predict(self.X_val)
        report = classification_report(self.y_val, pred, zero_division=0)

        print(report)

        return pipe, report