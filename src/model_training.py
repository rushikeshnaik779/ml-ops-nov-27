from sklearn.linear_model import LogisticRegression


def train_model(x, y): 
    """
    train_model :
        train the model. 
    
    Parameters: 
    --------------

        x: independent vairables 
        y : target variable 

    Returns : 
    ----------
        scikit-learn model (logistic model in this case)
    
    
    """
    model = LogisticRegression(solver='liblinear')

    model.fit(x,y)
    return model 