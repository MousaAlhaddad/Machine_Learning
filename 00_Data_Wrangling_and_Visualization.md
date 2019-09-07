# Data Wrangling and Visualization
## Importing Libraries
	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	import seaborn as sns
## Gathering Data
## Assessing Data
	df.head()
	df.describe()
	df.hist()
	sns.pairplot(df, hue="Column_Name")
	sns.heatmap(df.corr(), annot=True, cmap="YlGnBu")
## Cleaning Data 
	df.drop('Column_Name', axis = 1)
	df.map({"Old_Value_1":New_Value_1,"Old_Value_1":New_Value_2})

## Processing Data 
	 # Applying a logarithmic transformation on highly-skewed feature distributions
	 df[Skewed_Features_List].apply(lambda x: np.log(x + 1))
	 # Applying a scaling on numerical features
	 from sklearn.preprocessing import MinMaxScaler
	 scaler = MinMaxScaler()
	 scaler.fit_transform(df[Numerical_Features_List])
	 # Creating dummy variables
	 pd.get_dummies(df)
	 # Splitting for training and testing
	 from sklearn.cross_validation import train_test_split
	 X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size = 0.2, 
                                                    random_state = 0)
## Visualizing Data 

