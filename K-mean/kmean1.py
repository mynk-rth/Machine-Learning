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


    def showPlot(self):
        import matplotlib.pyplot as plt 
        color=['r','b']
        for i in range(self.k):
            plt.scatter(self.clusters[i],[i+1]*len(self.clusters[i]),c=color[i])

        plt.scatter(self.centroid,range(1,self.k+1),c='g',s=100,marker='*')
        plt.yticks(range(1,self.k+1))

        plt.show()
  




# -------------------------------------------------

model = Kmean(2)
data = [2, 4, 10, 12, 3, 20, 30, 11, 25]
model.fit(data)
model.show()

value = 30
model.predict(value)

model.showPlot()






# -------------------------------------------------


# import matplotlib.pyplot as plt

# plt.scatter(model.clusters[0], [1] * len(model.clusters[0]), c='r', label='Cluster 1')
# plt.scatter(model.clusters[1], [2] * len(model.clusters[1]), c='b', label='Cluster 2')

# plt.scatter(model.centroid, [1, 2], c='g', s=100, marker='*', label='Centroids')

# plt.xlabel("Data Values")
# plt.ylabel("Cluster")
# plt.yticks([1, 2], ["Cluster 1", "Cluster 2"])
# plt.legend()
# plt.grid(True)
# plt.show()






