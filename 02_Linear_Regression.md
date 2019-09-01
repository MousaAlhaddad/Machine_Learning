# Linear Regression
Fitting a linear regression model means finding the best line that fits the training data, a line that minimize either the mean absolute error or the mean squared error.
## Coding 
### Fitting the model 
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(x_values, y_values)
### Predicting new values 
    predicted _y_values = model.predict(unlabeled_x_values)
