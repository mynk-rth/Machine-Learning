import numpy as np
 
class Kmean:
    def __init__(self, k):
        self.k = k
        self.centroid = None
        self.clusters = None

    def fit(self, data):
        self.centroid = [4, 11]

        for i in range(100):
            clusters = [[] for i in range(self.k)]
            for v in data:
                distance = []
                for c in self.centroid:
                    dis = abs(v - c)
                    distance.append(dis)

                cluster_index = distance.index(min(distance))
                clusters[cluster_index].append(v)

            # print(clusters)

            nc = []                                    # new centriods
            for cluster in clusters:
                if len(cluster) > 0:
                    nc.append(sum(cluster) / len(cluster))
                else:
                    nc.append(0)
            # print(nc)

            if nc == self.centroid:
                break
            self.centroid = nc
        self.clusters = clusters
    
    def show(self):
        print('Clusters:')
        print(self.clusters)
        print()
        print('Centroid:')
        print(self.centroid)
        print()


    def predict(self, value):

        distance = []
        for centroid in self.centroid:
            dis = abs(value - centroid)
            distance.append(dis)

        cluster_index = distance.index(min(distance))
        print(value,"Belongs to Cluster",self.clusters[cluster_index])
        return self.clusters[cluster_index]
  




# -------------------------------------------------

model = Kmean(2)
data = [2, 4, 10, 12, 3, 20, 30, 11, 25]
model.fit(data)
model.show()

value = 30
model.predict(value)









