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
      
- The elbow method can be used to choose the best K in the K-means algorithm. 
- Choosing different centroids might result in different groupings for the same dataset.
- The best set of clusters is chosen through running the K-means algorithm many times and selecting the set with the
best score (the lowest average distance to the centroids). 
- Feature scaling is important in using the K-means algorithm. The most common way is the Z-Score Scaling [m=0,sd=1]. 
The second most common way is the Max-Min Scaling [0-1].
- 


