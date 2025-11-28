import pandas as pd
import numpy as np
import json

from sklearn.base import BaseEstimator
from helper.plots import feature_plots

from sklearn.metrics import mean_absolute_error, make_scorer, recall_score, precision_score, \
                                    f1_score, accuracy_score, classification_report, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

SEED = 42 

class DataPreprocessor:
    def __init__(self, data, columns):
        self.data = data
        self.columns = columns

    def get_missing(self):
        values = self.data.isna().sum().sort_values(ascending=False)
        values = values[values > 0]
        return values


    def get_missing_ratio(self):
        values = self.data.isna().mean().sort_values(ascending=False)
        values = values[values > 0]
        return values

    def check_multicollinearity(self, scaler, threshold):
        thr = threshold
        df_corr = self.data.copy()

        numeric_columns_corr = df_corr.select_dtypes(include=['int', 'float']).columns.to_list()
        numeric_columns = [col for col in numeric_columns_corr if numeric_columns_corr != self.columns]

        df_corr[numeric_columns_corr] = scaler.fit_transform(df_corr[numeric_columns_corr])

        correlation_matrix = df_corr[numeric_columns].corr()
        np.fill_diagonal(correlation_matrix.values, False)
        mask = (correlation_matrix > thr)

        fig = feature_plots.BuildHist(series_dict=None).matrix_multicollinearity(correlation_matrix, mask)

        return correlation_matrix, fig

class PipelineManager:
    def __init__(self, data, columns, model) -> None:
        self.data = data
        self.model = model
        self.columns = columns
        self.X, self.y = data.drop(columns=columns), data[columns]
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X, self.y, test_size=0.2, random_state=SEED)

    def create_pipeline(self, is_classification: bool) -> Pipeline:

        # Получаем числовые и категориальные колонки
        numeric_columns = self.X.select_dtypes(include=['int', 'float']).columns.to_list()
        categorical_columns = self.X.select_dtypes(exclude=['int', 'float']).columns.to_list()

        # Создание пайплайна для числовых данных
        numeric = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())])

        # Создание пайплайна для категориальных данных
        categorical = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        # Преобразование данных
        pre = ColumnTransformer(transformers=[
            ('num', numeric, numeric_columns),
            ('cat', categorical, categorical_columns)])

        # Выбор модели в зависимости от типа задачи
        if is_classification:
            pipeline = Pipeline([('prep', pre), ('clf', self.model)])
        else:
            pipeline = Pipeline([('prep', pre), ('reg', self.model)])

        return pipeline


    def make_pipeline(self) -> Pipeline:

        # Проверка на классификацию 
        is_classification = self.y.nunique() == 2
        return self.create_pipeline(is_classification), 'Classifier' if is_classification else 'Regression'
    
    def evaluate_model(self, model, label) -> None:

        # Подсчет метрик классификации 
        if label == 'Classifier':
            predict = model.predict(self.X_val)
            predict_proba = model.predict_proba(self.X_val)[:, 1]

            # Метрики
            accuracy = round(accuracy_score(self.y_val, predict), 2)
            recall = round(recall_score(self.y_val, predict), 2)
            precision = round(precision_score(self.y_val, predict), 2)
            f1_sr = round(f1_score(self.y_val, predict), 2)

            # Classification report
            class_report = classification_report(self.y_val, predict, output_dict=True)

            # Confusion matrix
            conf_matrix = confusion_matrix(self.y_val, predict)

            # ROC-кривая и AUC
            fpr, tpr, thresholds = roc_curve(self.y_val, predict_proba)
            curve = [fpr, tpr, thresholds]
            roc_auc = round(roc_auc_score(self.y_val, predict_proba), 2)

            #PR-кривая
            precision_curve, recall_curve, thresholds_curve = precision_recall_curve(self.y_val, predict)
            curve_pr = [precision_curve, recall_curve, thresholds_curve]

            metrics = [accuracy, recall, precision, f1_sr, class_report, conf_matrix, curve, roc_auc, curve_pr]

            print(f'{label} | Accuracy: {accuracy} | Recall: {recall} | Presicion: {precision} | F1_score: {f1_sr} | Roc-Auc: {roc_auc}')
            feature_plots.ClassificationPlot(metrics).metrics_plot()

        # Подсчет метрик регрессии
        else:      
            predict = np.expm1(model.predict(self.X_val))
            mae = round(mean_absolute_error(np.expm1(self.y_val), predict), 2)

            metrics = [mae]

            print(f'{label} | MAE: {mae}')

        return metrics, label

    def train_model(self, params, scorer=None) -> BaseEstimator:

        # Загрузка пайплайна
        pipeline, label = self.make_pipeline()

        # Кросс-валидация по данным
        cv = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True) if label == 'Classifier' else KFold(n_splits=5, random_state=SEED, shuffle=True)
        if scorer == None:
            scorer = make_scorer(f1_score, greater_is_better=False) if label == 'Classifier' else make_scorer(mean_absolute_error, greater_is_better=False)


        # Логарифмирование целевой метрики в задаче регрессии
        if label == 'Regression':
            self.y_train = np.log1p(self.y_train)
            self.y_val = np.log1p(self.y_val)

        # Подбор гиперпараметров
        search = RandomizedSearchCV(pipeline, 
                                    cv=cv, 
                                    param_distributions=params, 
                                    n_jobs=-1, 
                                    scoring=scorer, 
                                    error_score='raise', 
                                    verbose=1)
        
        # Обучение модели
        search.fit(self.X_train, self.y_train)

        print(f'Лучшие параметры модели: {search.best_params_}')

        # Запись лучшей модели
        best_search = search.best_estimator_
        metrics = self.evaluate_model(best_search, label)

        return best_search, metrics
    
    def imporance_columns(self) -> pd.DataFrame:
        # Получение важных признаков
        importance = permutation_importance(self.model, self.X_val, self.y_val, n_repeats=30, random_state=SEED, n_jobs=-1)

        # Отсортированный датафрейм важных признаков
        imp = pd.DataFrame(importance.importances_mean, index=self.X.columns, columns=['Importance']).sort_values(by=['Importance'], ascending=False)

        # Перевод научной записи в float
        imp['Importance'] = imp['Importance'].apply(lambda x: f'{x:.2f}')

        return imp
    

    def get_baseline(self):
        # Загрузка пайплайна
        baseline, label = self.make_pipeline()

        # Обучение модели
        baseline.fit(self.X_train, self.y_train)

        # Загрузка метрик у бейзлайна модели
        metrics, label = self.evaluate_model(baseline, label)

        return baseline, metrics, label
    
        
class Evaluate:
    """
    Класс для автоматической оценки влияния различных стратегий
    обработки признаков (feature engineering) на качество модели.

    Поддерживает:
    - выбор лучшей стратегии заполнения пропусков;
    - обработку выбросов;
    - оценку влияния удаления фичи;
    - оценку разницы метрик до и после обработки данных.
    """

    def __init__(self, data, columns, model):
        self.data = data
        self.columns = columns
        self.model = model

        # Бейзлайн-метрика "до" любых изменений
        self.baseline, self.metrics_before, self.label = PipelineManager(data, columns='Class', model=model).get_baseline()

# ----------------------------- Основные методы -----------------------------

    def run_strategies(self, strategies, index_labels, metric_name):
        """
        Запускает набор стратегий обработки фичи, сравнивает их влияние на метрику
        и применяет лучшую стратегию.

        Параметры
        ----------
        strategies : dict[str, callable]
            Словарь стратегий в формате: {'Название': функция_оценки}.
        index_labels : list[str]
            Список названий стратегий (кроме 'Baseline').
        metric_name : str
            Название метрики, например 'MAE' или 'F1-Score'.

        Возвращает
        ----------
        pandas.DataFrame
            DataFrame с результатами стратегий и выбранной лучшей.
        """

        # Создаем словарь для метрик, с заполненным нулем для бэйзлайна
        diffs = [0]
        for _, func in strategies.items():
            diffs.append(func())

        df = pd.DataFrame({f'Difference {metric_name}' : diffs,
                           f'Final {metric_name}' : [self.metrics_before[-1] + d for d in diffs]
                           }, index=['Baseline'] + index_labels)
                
        display(df) # type: ignore
        if self.label == 'Classifier':
            best = self.get_maxidx(df, metric_name)
        else:
            best = self.get_minidx(df, metric_name)
        #best_label = df.index[best]
        #data = self.evaluate_action(self.data, best_label)

        #return best 

    def select_best_fill_strategy(self):
        """
        Определяет и применяет лучшую стратегию заполнения пропусков
        (раздельно для категориальных и числовых фич).

        Возвращает
        ----------
        pandas.DataFrame
            Результаты применения стратегий и итоговое состояние данных.
        """

        if pd.api.types.is_object_dtype(self.data[self.columns]):
            strategies = {
                'Drop'      : self.evaluate_drop_impact, 
                'Fill'      : self.evaluate_fill_impact,
                'Mapping'   : self.evaluate_mapping_impact,
                'Bool'      : self.evaluate_bool_impact,
            }

            index_labels = ['Drop', 'Fill', 'Mapping', 'Bool']
        
        else:
            strategies = {
                'Drop'      : self.evaluate_drop_impact, 
                'Median'    : self.evaluate_fill_median,
                'Mean'      : self.evaluate_fill_mean,
                'Mode'      : self.evaluate_fill_mode,
            }

            index_labels = ['Drop', 'Median', 'Mean', 'Mode']

        return self.run_strategies(strategies, index_labels, metric_name='MAE')
    
    def select_best_outlier_strategy(self):
        """
        Определяет и применяет лучшую стратегию обработки выбросов.

        Для категориальных фич выводит предупреждение о необходимости
        предварительного преобразования в числовой формат.

        Возвращает
        ----------
        pandas.DataFrame
            Результаты применения стратегий и итоговое состояние данных.
        """

        if pd.api.types.is_object_dtype(self.data[self.columns]):
            print('Необходимо преобразовать в количественную фичу.')
            return self.data

        strategies = {
            'Drop'      : self.evaluate_drop_impact,
            'Outlier_5' : self.evaluate_five_outlier,
            'Outlier_10': self.evaluate_ten_outlier,
            'Outlier_20': self.evaluate_twenty_outlier,
        }
        index_labels = ['Drop', 'Outlier_5', 'Outlier_10', 'Outlier_20']

        return self.run_strategies(strategies, index_labels, metric_name='F1-Score')

    def get_minidx(self, data, metric_name):
        """
        Возвращает название лучшей стратегии на основе минимальной разницы метрик.

        Параметры
        ----------
        data : pandas.DataFrame
            Таблица со столбцом "Difference {metric_name}".
        metric_name : str
            Название метрики, например 'MAE' или 'F1-Score'.

        Возвращает
        ----------
        str
            Название лучшей стратегии.
        """

        col = f'Difference {metric_name}'
        return data[col][1:].idxmin()
    
    def get_maxidx(self, data, metric_name):
        """
        Возвращает название лучшей стратегии на основе максимальной разницы метрик.

        Параметры
        ----------
        data : pandas.DataFrame
            Таблица со столбцом "Difference {metric_name}".
        metric_name : str
            Название метрики, например 'MAE' или 'F1-Score'.

        Возвращает
        ----------
        str
            Название лучшей стратегии.
        """

        col = f'Difference {metric_name}'
        return data[col][1:].idxmax()

    def evaluate_action(self, data, best_label: str):
        """
        Применяет к данным действие, соответствующее лучшей найденной стратегии.

        Параметры
        ----------
        data : pandas.DataFrame
            Исходные данные.
        best_label : str
            Название выбранной стратегии.

        Возвращает
        ----------
        pandas.DataFrame
            Обновлённый набор данных после применения стратегии.
        """

        action = Action(data, self.columns)

        if pd.api.types.is_object_dtype(data[self.columns]):
            mapping = {
                'Drop'   : action.drop_data,
                'Fill'   : action.fill_data,
                'Mapping': action.mapping_data,
                'Bool'   : action.bool_data,
            }
        else:
            mapping = {
                'Drop'  : action.drop_data,
                'Median': action.median_data,
                'Mean'  : action.mean_data,
                'Mode'  : action.mode_data
            }

        func = mapping.get(best_label)
        if func is None:
            print(f'Нет действия для стратегии {best_label}, возвращаю исходные данные')
            return data

        return func()
    
    # ----------------------------- Стратегии для категориальных -----------------------------

    def evaluate_drop_impact(self):
        """
        Оценивает влияние удаления фичи на итоговую метрику модели.
        """

        if type(self.columns) == list:
            return np.nan

        data = self.data.copy()
        try:
            data = data.drop(columns=self.columns, axis=1)
        except Exception as e:
            print(e)

        difference = self.evaluate_difference(data)
        return difference

    def evaluate_fill_impact(self): 
        """
        Оценивает стратегию заполнения пропусков строкой 'No<Column>'.
        """

        if not pd.api.types.is_object_dtype(self.data[self.columns]):
            return np.nan

        data = self.data.copy()

        data[self.columns] = data[self.columns].fillna('No' + self.columns)

        difference = self.evaluate_difference(data)

        return difference

    def evaluate_mapping_impact(self):
        """
        Преобразует категориальные значения в числовые коды и оценивает влияние.
        """

        if not pd.api.types.is_object_dtype(self.data[self.columns]):
            return np.nan

        data = self.data.copy()
        try:
            data[self.columns] = data[self.columns].fillna('No' + self.columns)
            mapping = {}
            count = 0
            for value in data[self.columns].value_counts().index:
                if value not in mapping:
                    mapping[value] = count
                    count += 1
                else: 
                    continue
            data[self.columns] = data[self.columns].map(mapping)
        except Exception as e:
            print(f'Ошибка {e}')

        difference = self.evaluate_difference(data)
        return difference

    def evaluate_bool_impact(self):
        """
        Преобразует категориальную фичу в бинарный индикатор наличия значения.
        """

        if not pd.api.types.is_object_dtype(self.data[self.columns]):
            return np.nan
        data = self.data.copy()

        data['Has_'+ self.columns] = data[self.columns].notna().astype(int)
        data = data.drop(columns=[self.columns], axis=1)


        difference = self.evaluate_difference(data)
        return difference
    
    # ----------------------------- Стратегии для числовых -----------------------------

    def _evaluate_fill_with(self, how: str):
        """
        Универсальная функция для заполнения пропусков медианой, средним или модой.

        Параметры
        ----------
        how : str
            Метод заполнения ('median', 'mean', 'mode').

        Возвращает
        ----------
        float
            Разница метрик после применения стратегии.
        """

        data = self.data.copy()
        if how == 'median':
            value = data[self.columns].median()
        elif how == 'mean':
            value = data[self.columns].mean()
        elif how == 'mode':
            value = data[self.columns].mode()[0]
        else:
            raise ValueError(how)

        data[self.columns] = data[self.columns].fillna(value)
        return self.evaluate_difference(data)

    def evaluate_fill_median(self):
        """Заполняет пропуски медианой и оценивает влияние."""
        return self._evaluate_fill_with('median')

    def evaluate_fill_mean(self):
        """Заполняет пропуски средним и оценивает влияние."""
        return self._evaluate_fill_with('mean')

    def evaluate_fill_mode(self):
        """Заполняет пропуски модой и оценивает влияние."""
        return self._evaluate_fill_with('mode')
    
    # ----------------------------- Работа с выбросами -----------------------------
    
    def _evaluate_outlier(self, low_q, high_q):
        """
        Универсальная функция для удаления выбросов по квантилям.

        Параметры
        ----------
        low_q : float
            Нижняя граница квантиля.
        high_q : float
            Верхняя граница квантиля.

        Возвращает
        ----------
        float
            Разница метрик после применения фильтрации.
        """

        if not pd.api.types.is_numeric_dtype(self.data[self.columns]):
            return np.nan

        data = self.data.copy()
        q_low = data[self.columns].quantile(low_q)
        q_high = data[self.columns].quantile(high_q)


        data = data.loc[(data[self.columns] >= q_low) & (data[self.columns] <= q_high)]

        return self.evaluate_difference(data)

    def evaluate_five_outlier(self):
        """Удаляет выбросы за пределами 2.5% и 97.5% квантилей."""
        return self._evaluate_outlier(0.025, 0.975)

    def evaluate_ten_outlier(self):
        """Удаляет выбросы за пределами 5% и 95% квантилей."""
        return self._evaluate_outlier(0.05, 0.95)

    def evaluate_twenty_outlier(self):
        """Удаляет выбросы за пределами 10% и 90% квантилей."""
        return self._evaluate_outlier(0.1, 0.9)

    # ----------------------------- Метрики -----------------------------

    def evaluate_difference(self, data):
        """
        Считает разницу между текущей метрикой и бейзлайновой после изменения данных.
        """
        _, metrics_after, label = PipelineManager(data, columns='Class', model=self.model).get_baseline()
        for metric_before, metric_after in zip(self.metrics_before, metrics_after):
            difference = round(metric_after - metric_before, 2)

        return difference
    
    def evaluate_feature(self):
        """
        Оценивает, как удаление каждой фичи влияет на итоговую метрику модели.

        Возвращает
        ----------
        pandas.DataFrame
            Разница метрик для каждой фичи.
        """

        data = self.data.copy()
        eval_mae = []
        for col in self.columns:
            data = data.drop(columns=col)
            after_mae = PipelineManager(data, self.model).get_baseline()
            eval_mae.append((after_mae - self.metric_before))

        eval_df = pd.DataFrame(eval_mae, index=self.columns, columns=['Difference'])

        return eval_df
    



class Action():
    def __init__(self, data, columns):
        self.data = data
        self.columns = columns

    def drop_data(self):
        data = self.data.copy()

        try:
            data = data.drop(columns=[self.columns])
            print(f'Удалена фича {self.columns}')
        except:
            print(f'Фича {self.columns} уже удалена')
        return data

    def fill_data(self):
        data = self.data.copy()

        data[self.columns] = data[self.columns].fillna('No' + self.columns)
        print(f'Фича {self.columns} заполнена меткой')
        return data

    def mapping_data(self):
        data = self.data.copy()

        data[self.columns] = data[self.columns].fillna('No' + self.columns)
        mapping = {}
        count = 0
        for value in data[self.columns].value_counts().index:
            if value not in mapping:
                mapping[value] = count
                count += 1
            else: 
                continue
        data[self.columns] = data[self.columns].map(mapping)

        print(f'Применен мэппинг на фиче {self.columns}')
        return data

    def bool_data(self):
        data = self.data.copy()

        data['Has_'+ self.columns] = data[self.columns].notna().astype(int)
        data = data.drop(columns=[self.columns], axis=1)

        print(f'Фича преобразована в булевый тип {self.columns}')
        return data
    
    def median_data(self):
        data = self.data.copy()

        median = data[self.columns].median()
        data[self.columns] = data[self.columns].fillna(median)

        print(f'Фича заполнена медианой {self.columns}')
        return data

    def mean_data(self):
        data = self.data.copy()

        mean = data[self.columns].mean()
        data[self.columns] = data[self.columns].fillna(mean)

        print(f'Фича заполнена средним {self.columns}')
        return data

    def mode_data(self):
        data = self.data.copy()

        mode = data[self.columns].mode()[0]
        data[self.columns] = data[self.columns].fillna(mode)

        print(f'Фича заполнена модой {self.columns}')
        return data
    

def save_json(mae, path):
    data = {'MAE' : mae}

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ Данные сохранены в {path}")


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

