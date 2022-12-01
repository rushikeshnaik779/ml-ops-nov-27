import pandas as pd
import numpy as np 
import warnings
from imblearn.under_sampling import NearMiss

from sklearn.metrics import accuracy_score, roc_auc_score

warnings.filterwarnings("ignore")

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder




def data_train_val_test_split(data):

    """
    data_train_val_test_split: 
        spliting the data in train, test, val data.

    Parameters: 
    -----------
        data : pandas dataframe

    Returns: 
    --------
        train_df: pandas dataframe 
        val_df: pandas dataframe
        test_df: pandas dataframe

    """
    data.dropna(subset=['RainToday', 'RainTomorrow'], inplace=True)
    year = pd.to_datetime(data.Date).dt.year


    train_df = data[year<2015]
    val_df = data[year==2015]
    test_df = data[year>2015]


    return train_df, val_df, test_df

def imputing_missing_value(train_df, val_df, test_df): 
    """
    imputing_missing_value: 
        imputing missing value : SimpleImputer and LabelEncoder as imputer 
        to replace the categorical variable and numerical variable replace with mean values 

    Parameters 
    -----------
        train_df : pandas dataframe
        val_df : pandas dataframe
        test_df : pandas dataframe

    Returns 
    -------
        train_df : pandas dataframe
        val_df : pandas dataframe
        test_df : pandas dataframe 
    
    """
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
    """
    Data is imbalanced with 30 to  70 percent ratio. 
    we wil be undersampling it. 
    

    Parameters: 
    ----------
        train_df : pandas dataframe 

    
    Returns: 
    --------
    X, y the sampled data 
    
    """
    undersample = NearMiss(version=1, n_neighbors=3)

    x = train_df.drop('RainTomorrow', axis=1)
    y = train_df['RainTomorrow']
    x, y = undersample.fit_resample(x, y)

    return x, y 






