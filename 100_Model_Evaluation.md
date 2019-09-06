# Model Evaluation
## Coding
### Splitting for training and testing
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)
### Splitting for K-Fold cross calidation
    from sklearn.model_selection import KFold
    kf = KFold(data_size,test_size,shuffle=True)
### Calculating accuracy 
    # 1
    from sklearn.metrics import accuracy_score
    accuracy_score(y,y_pred)
    # 2
    sum(y==y_pred)/len(y)
### Calculating precision, recall and F1 score
    # Precision = True Positives/(True Positives + False Positives)
                = np.sum((preds == 1)&(actual == 1))/
                    (np.sum((preds == 1)&(actual == 1))+np.sum((preds == 1)&(actual == 0)))
    # Recall (sensitivity) = True Positives/(True Positives + False Negatives)
                = np.sum((preds == 1)&(actual == 1))/
                   (np.sum((preds == 1)&(actual == 1))+np.sum((preds == 0)&(actual == 1)))
    # F1 score is the weighted average of the precision and recall scores (2*Precision*Recall/(Precision+Recall))
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision_score(y_test,predictions)
    recall_score(y_test,predictions)
    f1_score(y_test,predictions)
### Calculating R2 score, mean squared error, and mean absolute error
    # sse = np.sum((actual-preds)**2)
    # sst = np.sum((actual-np.mean(actual))**2)
    # R2 score = 1 - sse/sst
    # MSE = np.mean((actual-preds)**2)
    # MAE = np.mean(abs(actual-preds))
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
### Drawing learning curves 
    train_sizes, train_scores, test_scores = learning_curve(
           estimator, X, y, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, num_trainings))
### Searching for the best hyperparameters 
    from sklearn.model_selection import GridSearchCV
    parameters = {'kernel':['poly', 'rbf'],'C':[0.1, 1, 10]}
    from sklearn.metrics import make_scorer
    from sklearn.metrics import f1_score
    scorer = make_scorer(f1_score)
    grid_obj = GridSearchCV(clf, parameters, scoring=scorer)
    grid_fit = grid_obj.fit(X, y)
    best_clf = grid_fit.best_estimator_

## Warnings
1. For classification problems that are skewed in their distributions, accuracy by itself is not a very good metric.
2. Never use testing data for training. Instead, add a cross validation set. 
