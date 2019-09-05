# Support Vector Machine
The algorithm works to minimize both the classification error and the margin error. 

## Coding 
### Fitting the model
	from sklearn.svm import SVC
	model = SVC() # Hyperparameters: C, kernel, degree, and gamma
	## The C parameter is for the classification error. 
	## The most common kernels are 'linear', 'poly', and 'rbf'. 
	## The degree parameter is linked to the polynomial kernel. 
	## The gamma parameter is linked to the RBF kernel. 
	model.fit(x_values, y_values)
### Warnings 
1. Large value of gamma tend to overfit. Small ones tend to underfit.
