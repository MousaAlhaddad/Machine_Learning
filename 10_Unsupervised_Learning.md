# Unsupervised Learning
- There are two types of unsupervised learning: clustering and dimensionality reduction.
## Clustering 
### K-means
      # Import KMeans and fit a model 
      from sklearn.cluster import KMeans
      algorithm = KMeans(K)
      model = algorithm.fit(data)
      labels = model.predict(data)
    
      # Score a model 
      model.score(data)
      
- The elbow method can be used to choose the best K in the K-means algorithm. Data visualizations can also help.
- Choosing different centroids might result in different groupings for the same dataset.
- The best set of clusters is chosen through running the K-means algorithm many times and selecting the set with the
best score (the lowest average distance to the centroids). 

      from sklearn import preprocessing as p
      p.StandardScaler().fit_transform(df)
      p.MinMaxScaler().fit_transform(df)
      
- Feature scaling is important in using the K-means algorithm. The most common way is the Z-Score Scaling [m=0,sd=1]. 
The second most common way is the Max-Min Scaling [0-1].
- The k-means algorithm has a spherical nature. Thus, it can not cluster two groups of a crescent shape or two groups of a ring shape.  


