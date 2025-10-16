import pandas as pd

def get_missing(data):
    values = data.isna().sum().sort_values(ascending=False)
    values = values[values > 0]
    return values


