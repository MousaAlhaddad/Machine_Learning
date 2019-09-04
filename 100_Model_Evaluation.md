# Model Evaluation
	Precision = True Positives/(True Positives + False Positives)
	Recall (sensitivity) = True Positives/(True Positives + False Negatives)
	F1 score is the weighted average of the precision and recall scores. 
## Coding
### Calculating accuracy 
    # 1
    from sklearn.metrics import accuracy_score
    accuracy_score(y,y_pred)
    # 2
    sum(y==y_pred)/len(y)
### Calculating precision, recall and f1_score
	from sklearn.metrics import precision_score, recall_score, f1_score
	precision_score(y_test,predictions)
	recall_score(y_test,predictions)
	f1_score(y_test,predictions)

		
## Warnings
1. For classification problems that are skewed in their distributions, accuracy by itself is not a very good metric.
