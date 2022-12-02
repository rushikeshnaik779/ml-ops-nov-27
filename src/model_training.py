from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier 
from xgboost import XGBClassifier




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

    elif type_model == 'xgb': 
        model = XGBClassifier()


    
    model.fit(x,y)
    return model 