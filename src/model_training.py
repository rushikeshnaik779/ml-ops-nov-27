from sklearn.linear_model import LogisticRegression


def train_model(x, y): 
    model = LogisticRegression(solver='liblinear')

    model.fit(x,y)
    return model 