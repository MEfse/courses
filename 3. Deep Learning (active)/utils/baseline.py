import pandas as pd
import numpy as np

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer

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
from sklearn.base import BaseEstimator, TransformerMixin
#from sklearn.feature_selection import 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from statsmodels.stats.outliers_influence import variance_inflation_factor

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import DataCollatorWithPadding

import os
from gensim.models import Word2Vec
import gensim.downloader as api
from transformers import BertTokenizer, AutoModelForSequenceClassification
from nltk.tokenize import WordPunctTokenizer

from tqdm.notebook import tqdm

import spacy # type: ignore
from multiprocessing import Pool



import logging

logging.basicConfig(level=logging.INFO, filename="./data/logging.log", filemode="w",
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
    

class SklearnPipelineManager():
    def __init__(self, vectozer, model):
        self.vectozer = vectozer
        self.model = model
        self.pipe = Pipeline([('vectorizer', self.vectozer), ('clf', self.model)]) 
    
    def fit(self, X_train, y_train):
        if self.pipe is None:
            self.build_pipeline()
        return self.pipe.fit(X_train, y_train)
    
    def predict(self, X_val):
        return self.pipe.predict(X_val)
    
class EmbeddingsManager():
    def __init__(self, 
                 vector_size=32, 
                 embedding_type='w2v',
                 glove_name='glove-twitter-100',
                 min_count=5, 
                 window=5, 
                 random_state=42,
                 clf=None):
        self.tokenazer = WordPunctTokenizer()

        self.embedding_type = embedding_type
        self.glove_name = glove_name

        self.vector_size = vector_size
        self.min_count = min_count
        self.window = window
        self.random_state = random_state

        self.emb = None
        self.clf = clf if clf is not None else LogisticRegression(max_iter=2000)
    
    def tokenaze(self, X):
        return [self.tokenazer.tokenize(str(x).lower()) for x in X]
     
    def fit_embeggings(self, X_train_tokens):
        if self.embedding_type == 'w2v':
            self.emb = Word2Vec(
                X_train_tokens,
                vector_size=self.vector_size,
                min_count=self.min_count,
                window=self.window,
                workers=4,
                seed=self.random_state
            ).wv
            self.vector_size = self.emb.vector_size

            return self
        
        if self.embedding_type == "glove":
            self.emb = api.load(self.glove_name)
            self.vector_size = self.emb.vector_size

            return self
        
        raise ValueError("Embedding_type must be 'w2v' or 'glove'")
        
    def featurize(self, X_tokens):
        X_vectors = []
        for tokens in X_tokens:
            vectors = [self.emb[token] for token in tokens if token in self.emb]
            
            if len(vectors) == 0:
                X_vectors.append(np.zeros(self.vector_size, dtype=np.float32))
            else:
                X_vectors.append(np.mean(vectors, axis=0))

        X_vectors = np.vstack(X_vectors)
        
        return X_vectors

    
    def fit(self, X_train_tokens, y_train):
        if self.emb is None:
            self.fit_embeggings(X_train_tokens)

        X_train_vec = self.featurize(X_train_tokens)
        self.clf.fit(X_train_vec, y_train)
        return self

    def predict(self, X_val_tokens):
        X_val_vec = self.featurize(X_val_tokens)
        pred = self.clf.predict(X_val_vec)
        return pred
    
    
class BertClassifierSimple(torch.nn.Module):
    def __init__(self, 
                 model_path='./models/bert',
                 num_classes=2,
                 batch_size=16, 
                 max_length=128,
                 lr=2e-5,
                 epochs=3
                 ):
        super().__init__()

        # Инициализация парамеров класса
        self.model_name = 'bert-base-uncased'
        self.model_path = model_path
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.max_length = max_length
        self.lr = lr
        self.epochs = epochs

        # Инициализация cpu/gpu
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Инициализация BERT, токеназера, оптимизатора и padding
        self.tokenazer_bert = BertTokenizer.from_pretrained(self.model_path, local_files_only=True)
        self.bert_model = AutoModelForSequenceClassification.from_pretrained(self.model_path, 
                                                                local_files_only=True, num_labels=2).to(self.device)
        self.optimizer = torch.optim.Adam(self.bert_model.parameters(), lr=2e-5)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenazer_bert)
        
        # mixed precision (AMP)
        #self.use_amp = True
        #self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Сохранение ошибок при обучении
        self.train_losses = []

    # ---utils---
    def _to_device(self, batch):
        return {k: v.to(self.device) for k, v in batch.items()}

    def _tokenaze_map(self, batch):
        return self.tokenazer_bert(batch['text'], 
                                   truncation=True, 
                                   padding=False, 
                                   max_length=self.max_length)
    
    # ---data---
    def dataloaders(self, X_train, y_train, X_val, y_val):
        train_ds = Dataset.from_dict({'text' : X_train.tolist(), 
                              'labels' : y_train.tolist()})
        val_ds = Dataset.from_dict({'text' : X_val.tolist(), 
                              'labels' : y_val.tolist()})
        
        train_ds = train_ds.map(self._tokenaze_map, batched=True, remove_columns=['text'])
        val_ds = val_ds.map(self._tokenaze_map, batched=True, remove_columns=['text'])

        col = ['labels', 'input_ids', 'token_type_ids', 'attention_mask']

        train_ds.set_format(type='torch', columns=col)
        val_ds.set_format(type='torch', columns=col)

        train_loader = DataLoader(train_ds, shuffle=True, batch_size=self.batch_size, collate_fn=self.data_collator)
        val_loader = DataLoader(val_ds, shuffle=False, batch_size=self.batch_size, collate_fn=self.data_collator)
                
        return train_loader, val_loader
    
    # ---train/eval---
    def fit(self, train_loader, val_loader=None):
        for epoch in range(1, self.epochs + 1):
            pbar = tqdm(train_loader, desc=f'{epoch} / {self.epochs}')
            self.bert_model.train()

            for batch in pbar:
                batch = self._to_device(batch)
                out = self.bert_model(**batch)
                out.loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.train_losses.append(out.loss.item())
                pbar.set_description(f'loss : {np.mean(self.train_losses[-100:])}')


    def evaluate(self, val_loader):
        self.bert_model.eval()
        eval_losses = []
        eval_preds = []
        eval_targets = []

        for batch in tqdm(val_loader, desc="eval"):
            batch = self._to_device(batch)
            with torch.no_grad():
                out = self.bert_model(**batch)
            eval_losses.append(out.loss.item())
            eval_preds.extend(out.logits.argmax(1).tolist())
            eval_targets.extend(batch['labels'].tolist())

            eval_loss = float(np.mean(eval_losses))
            accuracy = (np.array(eval_preds) == np.array(eval_targets)).mean()

        print(f"eval_loss={eval_loss:.4f} | accuracy={accuracy:.4f}")


    def predict_loader(self, data_loader):
        self.bert_model.eval()

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="predict"):
                batch = self._to_device(batch)
                out = self.bert_model(**batch)

                preds = out.logits.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch["labels"].cpu().numpy())

        return np.array(all_targets), np.array(all_preds)

