# Linear Regression
Fitting a linear regression model means finding the best line that fits the training data, a line that minimize either the mean absolute error or the mean squared error.

## Coding 
### Fitting the model 
    from sklearn.linear_model import LinearRegression
    model = LinearRegression() # normalize = True
    model.fit(x_values, y_values)
### Predicting new values 
    predicted_y_values = model.predict(unlabeled_x_values)
    
### Transforming into polynomial features
    from sklearn.preprocessing import PolynomialFeatures
    poly_feat = PolynomialFeatures(degree = n) # Change n
    X_poly = poly_feat.fit_transform(X) # X should be a numpy array of shape [n_samples, n_features] 
    ## Reshape the data using X.reshape(-1,1) or ,alternatively, np.array([[x] for x in X]) if needed
    poly_model = LinearRegression().fit(X_poly, y)
### Regularizing the model 
    ## Fitting a linear regression model while also using L1 regularization to control for model complexity
    from sklearn.linear_model import Lasso
    lasso_reg = Lasso()
    lasso_reg.fit(X,y)
    print(lasso_reg.coef_)
### Scaling features
    ## Standardizing 
    df["height_standard"] = (df["height"] - df["height"].mean()) / df["height"].std()
    ## Normalizing
    df["height_normal"] = (df["height"] - df["height"].min()) / (df["height"].max() - df['height'].min())
    ## Alternatively 
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
## Warnings
1. Linear regression produces a straight line model from the training data. Transform your training data, add more features, or use another kind of model if the relationship is not linear. 
2. Linear regression is sensitive to outliers.
3. A complex model might have a larger error than a simple one.
4. Feature scaling is important when using regularization. It is also necessary with distance based techniques such as Support Vector Machines and k-nearest neighbors.
