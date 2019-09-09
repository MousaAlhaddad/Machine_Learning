# Data Wrangling and Visualization


## Gathering Data
### Importing Libraries
	import numpy as np
	import pandas as pd
	import matplotlib.pyplot as plt
	import seaborn as sns
### Showing the python version
	from platform import python_version
	print(python_version())
	
	
## Assessing Data
	df.head()
	df.describe()
	df.hist()
	sns.pairplot(df, hue="Column_Name")
### Building a correlation matrix
	sns.heatmap(df.corr(), annot=True, cmap="YlGnBu")
### Getting the columns with no missing values 
	set(df.columns[~df.isnull().any()])
	
	
## Cleaning Data 
### Creating a new column containing the numbers of missing values of each row  
	df.isnull().sum(axis=1)
### Droping any row with a missing value
	df.dropna()
### Droping only the rows with all missing values
	df.dropna(axis=0, how='all')
### Droping only the rows with missing values in specific columns 
	df.dropna(subset=['Column_Name1', 'Colum_Name2'], how='any') 
### Dropping a column 
	df.drop('Column_Name', axis = 1)
### Filling null values by the mean of each column
	df.apply(lambda col: col.fillna(col.mean()), axis=0)
### Substituting each value in a Series with another value
	s.map({"Old_Value_1":New_Value_1,"Old_Value_1":New_Value_2})

	
## Processing Data 
### Applying a logarithmic transformation on highly-skewed feature distributions
	 df[Skewed_Features_List].apply(lambda x: np.log(x + 1))
	 # Applying a scaling on numerical features
	 from sklearn.preprocessing import MinMaxScaler
	 scaler = MinMaxScaler()
	 scaler.fit_transform(df[Numerical_Features_List])
### Creating dummy variables
	 pd.get_dummies(df)
### Encoding labels 
	from sklearn.preprocessing import LabelEncoder
	le = LabelEncoder()
	Column = le.fit_transform(Raw_Column)
	# Print one hot
	print(Column)
	# Reverse it with
	print(le.inverse_transform(Column))
### Splitting for training and testing
	 from sklearn.cross_validation import train_test_split
	 X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size = 0.2, 
                                                    random_state = 0)
						    
						    
## Visualizing Data 
### Coloring a pandas DataFrame column depending on its values
	df = pd.DataFrame({'Example':[0.5,0.6,0.7,-0.7,-0.2,0.1]})
	df.style.bar(subset=['Example'], align='mid', color=['#d65f5f', '#5fba7d'])

