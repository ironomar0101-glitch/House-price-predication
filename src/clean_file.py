import pandas as pd
import numpy as np

def drop_unnecessary_columns(df):
    return df.drop(columns=['society'], errors='ignore')

def drop_duplicates_data(df):
    return df.drop_duplicates().copy()

def impute_missing_values(df):
    na_columns = df.columns[df.isna().any()]
    for col in na_columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df
def convert_sqft_to_num(x):
    try:
        if isinstance(x, str) and '-' in x:
            tokens = x.split('-')
            return (float(tokens[0].strip()) + float(tokens[1].strip())) / 2
        return float(x)
    except:
        return None
def fix_total_sqft(df):

    df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)
    df = df.dropna(subset=['total_sqft'])
    return df

def run_cleaning_pipeline(df):
    df = drop_unnecessary_columns(df)
    df = impute_missing_values(df)
    df = fix_total_sqft(df)
    return df
