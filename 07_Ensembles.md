# Ensembles
By combining algorithms, we can often build models that perform better by minimizing bias and variance. 

## Codings 
### Fitting an AdaBoost model 
	from sklearn.ensemble import AdaBoostClassifier
	from sklearn.tree import DecisionTreeClassifier
	model = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=2), n_estimators = 4)
	model.fit(x_train, y_train)
### Fitting other models 
	BaggingClassifier
	RandomForestClassifier

## Warnings 
1. The default for most ensemble methods is a decision tree in sklearn.
2. Linear models have low variance, but high bias. An example of an algorithm that tends to have high variance and low bias is a decision tree.
