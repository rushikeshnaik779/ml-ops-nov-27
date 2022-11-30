import pandas as pd
import numpy as np 
import warnings
from imblearn.under_sampling import NearMiss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

warnings.filterwarnings("ignore")

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder


def data_extraction(data_path): 
    data = pd.read_csv(data_path)
    return data

def data_train_val_test_split(data):
    data.dropna(subset=['RainToday', 'RainTomorrow'], inplace=True)
    year = pd.to_datetime(data.Date).dt.year


    train_df = data[year<2015]
    val_df = data[year==2015]
    test_df = data[year>2015]


    return train_df, val_df, test_df

def imputing_missing_value(train_df, val_df, test_df): 

    numeric_cols = train_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = train_df.select_dtypes(include=object).columns.tolist()

    imputer = SimpleImputer(strategy='mean')
    encode = LabelEncoder()

    imputer.fit(train_df[numeric_cols])
    encode.fit(train_df['RainTomorrow'])

    train_df[numeric_cols] = imputer.transform(train_df[numeric_cols])
    val_df[numeric_cols] = imputer.transform(val_df[numeric_cols])
    test_df[numeric_cols] = imputer.transform(test_df[numeric_cols])

    train_df['RainTomorrow'] = encode.transform(train_df['RainTomorrow'])
    val_df['RainTomorrow'] = encode.transform(val_df['RainTomorrow'])
    test_df['RainTomorrow'] = encode.transform(test_df['RainTomorrow'])

    train_df['RainToday'] = encode.transform(train_df['RainToday'])
    val_df['RainToday'] = encode.transform(val_df['RainToday'])
    test_df['RainToday'] = encode.transform(test_df['RainToday'])

    obj = train_df.select_dtypes('object').columns.tolist()
    train_df = train_df.drop(obj, axis=1)
    val_df = val_df.drop(obj, axis=1)
    test_df = test_df.drop(obj, axis=1)

    return train_df, val_df, test_df
    

def under_sample(train_df):
    undersample = NearMiss(version=1, n_neighbors=3)

    x = train_df.drop('RainTomorrow', axis=1)
    y = train_df['RainTomorrow']
    x, y = undersample.fit_resample(x, y)

    return x, y 

def train_model(x, y): 
    model = LogisticRegression(solver='liblinear')

    model.fit(x,y)
    return model 

def predict(model, X): 
    return model.predict(X)

def metrics_cal(actuals, predicted): 
    metrics = {}
    metrics['auc'] = roc_auc_score(actuals, predicted)
    metrics['accuracy'] = accuracy_score(actuals, predicted)

    return metrics


if __name__ == "__main__": 
    path = '/Users/mohannaik/Desktop/Mohit_Project/ml-ops-nov-27/data/weatherAUS.csv'
    data = data_extraction(path)
    train_df, val_df, test_df = data_train_val_test_split(data)
    print(train_df.shape)
    train_df, val_df, test_df = imputing_missing_value(train_df, val_df, test_df)
    print(train_df.shape)
    x, y = under_sample(train_df)
    model = train_model(x, y)
    test_pred = predict(model, test_df.drop(['RainTomorrow'], axis=1))
    val_pred = predict(model, val_df.drop(['RainTomorrow'], axis=1))
    print(metrics_cal(test_df['RainToday'],test_pred))
    print(metrics_cal(val_df['RainToday'],val_pred))




