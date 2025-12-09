# Получение категорий фичей
try:
    categorical_columns = self.category_columns()[0]
    numeric_columns = self.category_columns()[1]

    categorical_columns = [col for col in categorical_columns if col != 'review_text']

    logging.info(f'Получены категории колонок: {categorical_columns} | {numeric_columns}.')
except Exception as e:
    logging.info(f'Не удалось получить категории колонок. Ошибка {e}.')


# Обработка пропущенных значений
try:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    num_imputer = SimpleImputer(strategy='mean')
    logging.info(f'Обработаны пропущенные значения.')
except Exception as e:
    logging.info(f'Не удалось обработать пропущенные значения. Ошибка {e}.')

# Преобразование данных
self.X_train['price_usd'] = np.log1p(self.X_train['price_usd'])
self.X_val['price_usd'] = np.log1p(self.X_val['price_usd'])
try:
    prep = ColumnTransformer(transformers=[
        ('num', StandardScaler(), [col for col in numeric_columns if col not in 
                                    ['rating', 'sentiment', 'verified_purchase', 'battery_life_rating', 
                                    'camera_rating', 'performance_rating', 'design_rating', 
                                    'display_rating', 'word_count', 'helpful_votes']]) ,
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ])
    logging.info(f'Данные преобразованы. Категориальные фичи через OHE, а количественные через StandardScaler.')
except Exception as e:
    logging.info(f'Не удалось преобразовать данные. Ошибка {e}.')