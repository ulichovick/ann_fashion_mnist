def evaluate(model, X_test, Y_test):
    """
    evaluate the model
    """
    print(model.evaluate(X_test, Y_test))