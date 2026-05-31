def train(model, X_train, y_train):
    model.fit(X_train, y_train)


def predict(model, X_test):
    return model.predict(X_test)
