from data_extraction import data_extraction
from data_preparation import data_train_val_test_split, imputing_missing_value, under_sample
from evaluation import metrics_cal, predict
from model_training import train_model


if __name__ == "__main__": 
    path = '../data/weatherAUS.csv'
    data = data_extraction(path)
    train_df, val_df, test_df = data_train_val_test_split(data)
    
    train_df, val_df, test_df = imputing_missing_value(train_df, val_df, test_df)
    
    x, y = under_sample(train_df)
    model = train_model(x, y)
    test_pred = predict(model, test_df.drop(['RainTomorrow'], axis=1))
    val_pred = predict(model, val_df.drop(['RainTomorrow'], axis=1))
    
    
    print(metrics_cal(test_df['RainToday'],test_pred))
    print(metrics_cal(val_df['RainToday'],val_pred))