import pandas as pd
import numpy as np
import json
from helper.plots import feature_plots


from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer
from sklearn.model_selection import KFold, train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

SEED = 42 

class DataPreprocessor:
    def __init__(self, data):
        self.data = data

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
        numeric_columns = [col for col in numeric_columns_corr if numeric_columns_corr != 'SalePrice']

        #categorical_columns_corr = df_corr.select_dtypes(exclude=['int', 'float']).columns.to_list()

        df_corr[numeric_columns_corr] = scaler.fit_transform(df_corr[numeric_columns_corr])

        correlation_matrix = df_corr[numeric_columns].corr()
        np.fill_diagonal(correlation_matrix.values, False)
        mask = (correlation_matrix > thr)

        fig = feature_plots.BuildHist(series_dict=None).matrix_multicollinearity(correlation_matrix, mask)

        return correlation_matrix, fig

class PipelineManager:
    def __init__(self, data, model):
        self.data = data
        self.model = model
        self.X, self.y = data.drop(columns=['SalePrice']), data['SalePrice']
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X, self.y, test_size=0.2, random_state=SEED)
        self.y_train = np.log1p(self.y_train)
        self.y_val = np.log1p(self.y_val)

    def make_pipeline(self):
        
        numeric = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())])

        categorical = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        numeric_columns = self.X_train.select_dtypes(include=['int', 'float']).columns.to_list()
        categorical_columns= self.X_train.select_dtypes(exclude=['int', 'float']).columns.to_list()

        pre = ColumnTransformer(transformers=[
            ('num', numeric, numeric_columns),
            ('cat', categorical, categorical_columns)])

        pipeline = Pipeline([
            ('prep', pre),
            ('reg', self.model)
        ])

        return pipeline


    def train_model(self, params, grid=False):

        pipeline = self.make_pipeline()

        cv = KFold(n_splits=5, random_state=SEED, shuffle=True)

        scorer = make_scorer(mean_absolute_error, greater_is_better=False)

        search = RandomizedSearchCV(pipeline, 
                                    cv=cv, 
                                    param_distributions=params, 
                                    n_jobs=-1, 
                                    scoring=scorer, 
                                    error_score='raise', 
                                    verbose=1)
        

        search.fit(self.X_train, self.y_train)

        print(f'Лучшие параметры модели: {search.best_params_}')

        best_forest = search.best_estimator_
        forest_predict = np.expm1(best_forest.predict(self.X_val))
        forest_mae = round(mean_absolute_error(np.expm1(self.y_val), forest_predict), 2)

        print(f'MAE модели: {forest_mae}')

        return best_forest
    
    def imporance_columns(self):
        importance = permutation_importance(self.model, self.X_val, self.y_val, n_repeats=30, random_state=SEED, n_jobs=-1)
        imp = pd.DataFrame(importance.importances_mean, index=self.X.columns, columns=['Importance']).sort_values(by=['Importance'], ascending=False)
        imp['Importance'] = imp['Importance'].apply(lambda x: f'{x:.2f}')

        return imp
    

    def get_baseline(self):

        baseline = self.make_pipeline()
        baseline.fit(self.X_train, self.y_train)

        y_pred = np.expm1(baseline.predict(self.X_val))
        mae = round(mean_absolute_error(np.expm1(self.y_val), y_pred), 2)

        return mae
        

class Evaluate:
    def __init__(self, train, test, columns, model):
        self.train = train
        self.test = test
        self.columns = columns
        self.model = model
        self.mae_before = PipelineManager(self.train, self.model).get_baseline()

    def info(self):
        mae_drop = self.evaluate_drop_impact() 
        if pd.api.types.is_object_dtype(self.train[self.columns]):   
            mae_fill    = self.evaluate_fill_impact()     
            mae_mapping = self.evaluate_mapping_impact()     
            mae_bool    = self.evaluate_bool_impact()  

            values = [0, mae_drop, mae_fill, mae_mapping, mae_bool]

        else:
            mae_median  = self.evaluate_fill_median()
            mae_mean    = self.evaluate_fill_mean()
            mae_mode    = self.evaluate_fill_mode()

            values = [0, mae_drop, mae_median, mae_mean, mae_mode]

        data_evaluate = self.get_plot(values)
        display(data_evaluate) # type: ignore

        best = self.get_minidx(data_evaluate)
        train = self.evaluate_action(self.train, best)
        test = self.evaluate_action(self.test, best)

        return train, test

    def get_minidx(self, data_evaluate):
        return data_evaluate['Difference MAE'][1:].idxmin()

    def get_plot(self, values):
        if pd.api.types.is_object_dtype(self.train[self.columns]):
            self.index = ['Baseline', 'Drop', 'Fill', 'Mapping', 'Bool'] 
        else:
            self.index = ['Baseline', 'Drop', 'Median', 'Mean', 'Mode']

        data_evaluate = pd.DataFrame({
            'Difference MAE': values,
            'Final MAE': [self.mae_before,
                            self.mae_before + values[1],
                            self.mae_before + values[2],
                            self.mae_before + values[3],
                            self.mae_before + values[4]]}, index=self.index) 

        return data_evaluate


    def evaluate_action(self, data, best):
        if pd.api.types.is_object_dtype(data[self.columns]):
            if best == self.index[1]:
                data = Action(data, self.columns).drop_data()
                return data
            elif best == self.index[2]:
                data = Action(data, self.columns).fill_data()
                return data
            elif best == self.index[3]:
                data = Action(data, self.columns).mapping_data()
                return data
            elif best == self.index[4]:
                data = Action(data, self.columns).bool_data()
                return data
            
        else:
            if best == self.index[1]:
                data = Action(data, self.columns).drop_data()
                return data
            elif best == self.index[2]:
                data = Action(data, self.columns).median_data()
                return data
            elif best == self.index[3]:
                data = Action(data, self.columns).mean_data()
                return data
            elif best == self.index[4]:
                data = Action(data, self.columns).mode_data()
                return data

    def evaluate_drop_impact(self):
        if type(self.columns) == list:
            return np.nan

        data = self.train.copy()
        try:
            data = data.drop(columns=self.columns, axis=1)
        except Exception as e:
            print(e)

        difference = self.evaluate_difference(data)
        return difference

    def evaluate_fill_impact(self):           
        if not pd.api.types.is_object_dtype(self.train[self.columns]):
            return np.nan

        data = self.train.copy()
        try:
            data[self.columns] = data[self.columns].fillna('No' + self.columns)
        except Exception as e:
            print(e)

        difference = self.evaluate_difference(data)
        return difference

    def evaluate_mapping_impact(self):
        if not pd.api.types.is_object_dtype(self.train[self.columns]):
            return np.nan

        data = self.train.copy()
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
        if not pd.api.types.is_object_dtype(self.train[self.columns]):
            return np.nan
        data = self.train.copy()
        try:
            data['Has_'+ self.columns] = data[self.columns].notna().astype(int)
            data = data.drop(columns=[self.columns], axis=1)
        except Exception as e:
            print(f'Ошибка {e}')

        difference = self.evaluate_difference(data)
        return difference

    def evaluate_fill_median(self):
        data = self.train.copy()

        median = data[self.columns].median()
        data[self.columns] = data[self.columns].fillna(median)
        difference = self.evaluate_difference(data)

        return difference

    def evaluate_fill_mean(self):
        data = self.train.copy()

        mean = data[self.columns].mean()
        data[self.columns] = data[self.columns].fillna(mean)
        difference = self.evaluate_difference(data)

        return difference

    def evaluate_fill_mode(self):
        data = self.train.copy()
        
        mode = data[self.columns].mode()[0]
        data[self.columns] = data[self.columns].fillna(mode)
        difference = self.evaluate_difference(data)

        return difference

    def evaluate_difference(self, data):

        mae_after = PipelineManager(data, self.model).get_baseline()
        difference = round(mae_after - self.mae_before, 2)

        return difference
    
    def evaluate_feature(self):
        data = self.train.copy()
        eval_mae = []
        for col in self.columns:
            data = data.drop(columns=col)
            after_mae = PipelineManager(data, self.model).get_baseline()
            eval_mae.append((after_mae - self.mae_before))

        eval_df = pd.DataFrame(eval_mae, index=self.columns, columns=['Difference MAE'])

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

