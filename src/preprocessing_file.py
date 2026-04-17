import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def transform_skewed_features(df):
    df['price'] = np.log1p(df['price'])
    df['total_sqft'] = np.log1p(df['total_sqft'])
    return df

def clean_size_feature(df):
    df['size'] = df['size'].apply(lambda x: int(str(x).split(' ')[0]))
    df = df[df['size'] <= 10]
    return df

def encode_categorical_features(df):
    df['availability'] = df['availability'].apply(
        lambda x: 1 if x in ['Ready To Move', 'Immediate Possession'] else 0
    )

    le = LabelEncoder()
    df['area_type'] = le.fit_transform(df['area_type'])

    if 'location' in df.columns:
        location_stats = df['location'].value_counts()
        location_stats_less_than_10 = location_stats[location_stats <= 10]
        df['location'] = df['location'].apply(
            lambda x: 'other' if x in location_stats_less_than_10 else x
        )
        df = pd.get_dummies(df, columns=['location'], drop_first=True)

    return df

def run_preprocessing_pipeline(df):
    df = clean_size_feature(df)
    df = transform_skewed_features(df)
    df = encode_categorical_features(df)

    X = df.drop(columns='price')
    y = df['price']
    return X, y
