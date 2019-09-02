# Linear Regression
Fitting a linear regression model means finding the best line that fits the training data, a line that minimize either the mean absolute error or the mean squared error.
## Coding 
### Fitting the model 
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(x_values, y_values)
### Predicting new values 
    predicted_y_values = model.predict(unlabeled_x_values)
    
### Transforming into polynomial features
    from sklearn.preprocessing import PolynomialFeatures
    poly_feat = PolynomialFeatures(degree = n) # Change n
    X_poly = poly_feat.fit_transform(X) # X should be a numpy array of shape [n_samples, n_features] 
    ## Reshape the data using X.reshape(-1,1) or ,alternatively, np.array([[x] for x in X]) if needed
    poly_model = LinearRegression().fit(X_poly, y)

## Warnings
1. Linear regression produces a straight line model from the training data. Transform your training data, add more features, or use another kind of model if the relationship is not linear. 
2. Linear regression is sensitive to outliers.
