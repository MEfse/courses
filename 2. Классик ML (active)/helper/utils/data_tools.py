import pandas as pd
import numpy as np
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

        categorical_columns_corr = df_corr.select_dtypes(exclude=['int', 'float']).columns.to_list()

        df_corr[numeric_columns_corr] = scaler.fit_transform(df_corr[numeric_columns_corr])

        correlation_matrix = df_corr[numeric_columns].corr()
        np.fill_diagonal(correlation_matrix.values, False)
        mask = (correlation_matrix > thr)

        fig = feature_plots.BuildHist(series_dict=None).matrix_multicollinearity(correlation_matrix, mask)

        return correlation_matrix, fig

class PipelineManager:
    def __init__(self, X, y, model):
        self.X = X
        self.y = y
        self.model = model

    def data_split(self, test_size=0.2):
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, shuffle=True, test_size=test_size, random_state=SEED)
        return X_train, X_val, y_train, y_val

    def make_pipeline(self):

        numeric = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())])

        categorical = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('scaler', OneHotEncoder(handle_unknown='ignore'))])

        numeric_columns = self.X.select_dtypes(include=['int', 'float']).columns.to_list()
        categorical_columns= self.X.select_dtypes(exclude=['int', 'float']).columns.to_list()

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

        if grid == True:
            search = GridSearchCV(pipeline, 
                                    cv=cv, 
                                    param_distributions=params, 
                                    n_jobs=-1, 
                                    scoring=scorer)
        else:
            search = RandomizedSearchCV(pipeline, 
                                        cv=cv, 
                                        param_distributions=params, 
                                        n_jobs=-1, 
                                        scoring=scorer)

        

        search.fit(self.X, self.y)


        print(f'Лучшие параметры модели: {search.best_params_}')
        #print(f'MAE модели: {round((mean_absolute_error(np.expm1())), 2)}')

        return search

    def imporance_columns(self):
        importance = permutation_importance(self.model, self.X, self.y, n_repeats=30, random_state=SEED, n_jobs=-1)
        imp = pd.DataFrame(importance.importances_mean, index=self.X.columns, columns=['Importance']).sort_values(by=['Importance'], ascending=False)
        imp['Importance'] = imp['Importance'].apply(lambda x: f'{x:.2f}')

        return imp




