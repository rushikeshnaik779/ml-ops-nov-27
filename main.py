from src.data_extraction import data_extraction
from src.data_preparation import data_train_val_test_split, imputing_missing_value, under_sample
from src.evaluation import metrics_cal, predict
from src.model_training import train_model

import mlflow 

if __name__ == "__main__": 
    mlflow_experiment_path = "./artifacts"
    mlflow.set_experiment(mlflow_experiment_path)

    path = 'data/weatherAUS.csv'
    data = data_extraction(path)
    print(data.shape)
    train_df, val_df, test_df = data_train_val_test_split(data)
    
    train_df, val_df, test_df = imputing_missing_value(train_df, val_df, test_df)
    
    x, y = under_sample(train_df)
    with mlflow.start_run(): 
        mlflow.log_param('Author', 'Rushikesh')
        model = train_model(x, y, type_model='rf')

        mlflow.sklearn.log_model( model, 'rf')

        test_pred = predict(model, test_df.drop(['RainTomorrow'], axis=1))
        val_pred = predict(model, val_df.drop(['RainTomorrow'], axis=1))
        auc_test, accuracy_test = metrics_cal(test_df['RainToday'],test_pred)
        
        mlflow.log_metric('AUC', auc_test )
        mlflow.log_metric('Accuracy', accuracy_test)
        print(metrics_cal(test_df['RainToday'],test_pred))
        print(metrics_cal(val_df['RainToday'],val_pred))