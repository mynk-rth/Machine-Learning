import numpy as np
 
class Kmean:
    def __init__(self, k):
        self.k = k
        self.centroid = None
        self.clusters = None

    def fit(self, data):
        
        data = np.array(data,dtype=float)
        self.centroid = data[:self.k]

        for i in range(100):
            clusters = [[] for i in range(self.k)]
            for v in data:
                distance = []
                for c in self.centroid:
                    dis = np.sqrt(np.sum((v - c) ** 2))
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

            if np.allclose(nc, self.centroid):
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
        value=np.array(value,float)
        distance = []
        for centroid in self.centroid:
            dis = np.sqrt(np.sum(value-centroid)**2)
            distance.append(dis)

        cluster_index = distance.index(min(distance))
        print(value,"Belongs to Cluster",self.clusters[cluster_index])
        return self.clusters[cluster_index]


    def showPPlot(self):
        import matplotlib.pyplot as plt 
        color=['r','b']
 
        for i in range(self.k):
            cluster = np.array(self.clusters[i])
            plt.scatter(cluster[:, 0],cluster[:, 1],c=color[i])

        centroid = np.array(self.centroid)
        plt.scatter(centroid[:,0],centroid[:,1],marker="X",s=200,c='g')

        plt.show()



    




# -------------------------------------------------

model = Kmean(2)
data = [[2, 10],[2, 5],[8, 4],[5, 8],[7, 5],[6, 4],[1, 2],[4, 9]]
model.fit(data)
model.show()

value = [3, 7]
p = model.predict(value)

model.showPlot()
model.showPPlot()



















# -------------------------------------------------


    # def showPlot(self):

    #     import matplotlib.pyplot as plt
    #     import numpy as np

    #     for i, cluster in enumerate(self.clusters):

    #         cluster = np.array(cluster)

    #         if len(cluster) > 0:
    #             plt.scatter(
    #             cluster[:, 0],
    #             cluster[:, 1],
    #             label=f"Cluster {i + 1}"
    #             )

    #     # Plot centroids
    #     centroids = np.array(self.centroid)

    #     plt.scatter(
    #         centroids[:, 0],
    #         centroids[:, 1],
    #         marker="X",
    #         s=200,
    #         c="red",
    #         label="Centroids"
    #         )

    #     plt.xlabel("X")
    #     plt.ylabel("Y")
    #     plt.title("K-Means Clustering")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()