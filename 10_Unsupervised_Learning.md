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

### Agglomerative Hierarchical Clustering
      from sklearn.cluster import AgglomerativeClustering
      algorithm = AgglomerativeClustering(n_clusters=K, linkage='ward')
      labels = algorithm.fit_predict(data)
      
- The parameter linkage is optional {“ward” (default), “complete”, “average”, “single”}. The linkage criterion determines which distance to use between sets of observation. The algorithm will merge the pairs of cluster that minimize this criterion. **ward** minimizes the variance of the clusters being merged. **average** uses the average of the distances of each observation of the two sets. **complete** or maximum linkage uses the maximum distances between all observations of the two sets. **single** uses the minimum of the distances between all observations of the two sets. 
- Clustering after normalization using the **ward** linkage criterion might increase the accuracy. 
- The **single** linkage is more prone to result in elongated shapes.

      # Plot the hierarchical clustering as a dendrogram.
      from scipy.cluster.hierarchy import linkage, dendrogram
      linkage_matrix = linkage(normalized_X, linkage_type)
      dendrogram(linkage_matrix)
      
- The dendrogram illustrates how each cluster is composed by drawing a U-shaped link between a cluster and its children. The top of the U-link indicates a cluster merge. The two legs of the U-link indicate which clusters were merged. The length of the two legs of the U-link represents the distance between the child clusters.
      
      # Plot a matrix dataset as a hierarchically-clustered heatmap
      sns.clustermap(normalized_X, method=linkage_type)

- A clustermap is a detailed dendrogram which also visualizes the dataset in more detail.
- Hierarchical Clustering is sensitive to **outliers** and **computationally expensive**.

### DBSCAN 
      from sklearn.cluster import DBSCAN
      model = DBSCAN(eps=0.5, min_samples=5)
      labels = model.fit_predict(data)
      
- Density-Based Spatial Clustering of Applications with Noise finds core samples of high density and expands clusters from them. 
- **eps** (float, optional) is the maximum distance between two samples for one to be considered as in the neighborhood of the other. 
- **min_samples** (int, optional) is the number of samples in a neighborhood for a point to be considered as a core point. This includes the point itself. 
- DBSCAN does not need specifying the number of clusters.

### GaussianMixture
      from sklearn.mixture import GaussianMixture
      model = GaussianMixture(n_components=K)
      model.fit(data)
      model.predict(data)

###  Clustering Metrics 
      from sklearn.metrics import adjusted_rand_score
      adjusted_rand_score(labels_true, labels_pred)

- The Rand Index computes a similarity measure between two clusterings between -1.0 and 1.0. Random labelings have an ARI close to 0.0. 1.0 stands for a perfect match.

      from sklearn.metrics import silhouette_score
      silhouette_score(data,labels)
      
- The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters. Negative values generally indicate that a sample has been assigned to the wrong cluster, as a different cluster is more similar.

## Dimensionality Reduction
### PCA 
      from sklearn.decomposition import PCA
      pca = PCA(n_components)
      X_pca = pca.fit_transform(X)
      
      pca.explained_variance_ratio_
      
- Make sure tor scale the data before using PCA.
- The aim of PCA is to maximize variance.

### Random Projection
      from sklearn.random_projection import SparseRandomProjection
      rp = SparseRandomProjection(n_components='auto',eps=0.1)
      new_x = rp.fit_transform(x)
      
- n_components (int or ‘auto’, optional (default = ‘auto’)) can be automatically adjusted according to the number of samples in the dataset.
- eps (strictly positive float, optional, (default=0.1)) with smaller values lead to better embedding and higher number of dimensions (n_components) in the target projection space.

### ICA 
      X = list(zip(data))
      from sklearn.decomposition import FastICA
      ica = FastICA(n_components)
      X_ica = pca.fit_transform(X)










