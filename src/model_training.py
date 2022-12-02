from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 



def train_model(x, y, type_model='lr'): 
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
    if type_model=='lr':
        model = LogisticRegression(solver='liblinear')
    elif type_model == 'rf':
        model = RandomForestClassifier(max_depth = 10, 
        random_state=7)


    
    model.fit(x,y)
    return model 