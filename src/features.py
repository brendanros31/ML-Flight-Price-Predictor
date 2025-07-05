import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler



# Encoding selected Features
def encode_features(df, categorical_cols):
    le = LabelEncoder()

    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
    return df



# Scaling data
def scale_features(X_split):
    scaler = StandardScaler()
    return scaler.fit_transform(X_split)


