from sklearn.metrics import accuracy_score, roc_auc_score


def predict(model, X): 
    return model.predict(X)

def metrics_cal(actuals, predicted): 
    metrics = {}
    metrics['auc'] = roc_auc_score(actuals, predicted)
    metrics['accuracy'] = accuracy_score(actuals, predicted)

    return metrics