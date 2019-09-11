# Neural Networks

### Arranging the data  
	import keras
	features = np.array(features))
	targets = np.array(keras.utils.to_categorical(targets, 2))

### Creating a Sequential model
	from keras.models import Sequential
	from keras.layers.core import Dense, Activation
	model = Sequential()	
	# Adding an input layer of 32 nodes 
	model.add(Dense(32, input_dim=))
	# Adding a softmax activation layer
	model.add(Activation('softmax')) # others: relu
	# Adding a fully connected output layer
	model.add(Dense(1))
	# Adding a sigmoid activation layer
	model.add(Activation('sigmoid'))
	# Compiling the model  
	from keras.optimizers import SGD
	model.compile(loss="categorical_crossentropy", optimizer="adam", metrics = ["accuracy"]) 
		## loss= binary_crossentropy,  mean_squared_error
		## optimizer: 'rmsprop' (RMS stands for Root Mean Squared Error), 
			"adam" (Adaptive Moment Estimation), or SGD() (Stochastic Gradient Descent)
	# Showing results  
	model.summary()
	
### Fitting the model 
	model.fit(X, y, nb_epoch=1000,  batch_size=100, verbose=0)
	
### Evaluating the model 
	model.evaluate(features, targets)

