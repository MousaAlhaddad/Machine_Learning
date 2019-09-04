# Naive Bayes 
Spam detection is one of the major applications of the Naive Bayes algorithm.

## Coding 
### Calculating probabilities through theorem
	def P(C,pC, nN,ca,no,po,ne):
    return {ca:C, 
        no:1-C,
        
        po+ca:pC,
        ne+ca: 1-pC,
        po+no: 1-nN, 
        ne+no:nN,
        
        ca+"&"+po:C*pC,
        ca+"&"+ne:C*(1-pC), 
        no+"&"+po:(1-C)*(1-nN),
        no+"&"+ne:(1-C)*nN,
        
        po:C*pC+(1-C)*(1-nN), 
        ne:C*(1-pC)+(1-C)*(nN),
        
        ca+po: C*pC/(C*pC+(1-C)*(1-nN)),
        no+po:1-C*pC/(C*pC+(1-C)*(1-nN)),
        ca+ne:C*(1-pC)/(C*(1-pC)+(1-C)*(nN)),
        no+ne:1-C*(1-pC)/(C*(1-pC)+(1-C)*(nN))}
### Counting word frequencies 
	from sklearn.feature_extraction.text import CountVectorizer
	count_vector = CountVectorizer(lowercase=True,stop_words=None,token_pattern='(?u)\\b\\w\\w+\\b')
	training_data = count_vector.fit_transform(X_train)
	testing_data = count_vector.transform(X_test)
	count_vector.get_feature_names()
### Fitting the model 
	from sklearn.naive_bayes import MultinomialNB
	naive_bayes = MultinomialNB()
	naive_bayes.fit(training_data,y_train)
## Warnings 
1. The multinomial Naive Bayes algorithm is suitable for classification with discrete features (such as word counts for text classification). It takes in integer word counts as its input. On the other hand, Gaussian Naive Bayes is better suited for continuous data as it assumes that the input data has a Gaussian (normal) distribution.
