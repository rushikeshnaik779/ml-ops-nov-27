from sklearn.metrics import accuracy_score, roc_auc_score


def predict(model, X): 
    """
    predict : 
        it will generate the predictions based on the model 
        we provided in the functions

    Parameters: 
    -----------
        model: 
            sklearn model 
        X : 
            pandas dataframe 

    Returns: 
        predicted arary 
    """
    return model.predict(X)

def metrics_cal(actuals, predicted): 
    """
    Metrics_cal: function will generate dictionary of output with metrics focusing on as AUC and accuracy 

    Parameters:  
    ----------
        actual : series 
        predicted : series 

    Returns: 
    ---------
        Dictionary
    
    """
    metrics = {}
    metrics['auc'] = roc_auc_score(actuals, predicted)
    metrics['accuracy'] = accuracy_score(actuals, predicted)

    return metrics['auc'], metrics['accuracy']