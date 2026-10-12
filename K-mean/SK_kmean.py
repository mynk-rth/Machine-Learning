import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = [[2, 10],[2, 5],[8, 4],[5, 8],[7, 5],[6, 4],[1, 2],[4, 9]]
data = np.array(data)



model = KMeans(n_clusters=2)

model.fit(data)

labels = model.labels_
print("Cluster Lables : ")
print(labels)

centroids = model.cluster_centers_
print("\n Centroids : ")
print(centroids)

plt.scatter(
    data[:,0],
    data[:,1],
    c=labels)


plt.scatter(centroids[:,0],
centroids[:,1],
marker="X",s=200)


plt.xlabel("X")
plt.ylabel("Y")
plt.title("K-means Clustering")
plt.show()