# Model Evaluation
## Coding
### Splitting for training and testing
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)
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
## Warnings
1. For classification problems that are skewed in their distributions, accuracy by itself is not a very good metric.