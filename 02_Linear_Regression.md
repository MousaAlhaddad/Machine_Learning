# Linear Regression
Fitting a linear regression model means finding the best line that fits the training data, a line that minimize either the mean absolute error or the mean squared error.
## Coding 
### Fitting the model 
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(x_values, y_values)
### Predicting new values 
    predicted_y_values = model.predict(unlabeled_x_values)
## Warnings
1. Linear regression produces a straight line model from the training data. Transform your training data, add more features, or use another kind of model if the relationship is not linear. 
2. Linear regression is sensitive to outliers.
