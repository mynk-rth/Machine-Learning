# import math

# class Kmean:

#     def __init__(self, k):
#         self.k = k
#         self.centroid = None
#         self.clusters = None


#     def fit(self, data):

#         self.centroid = [
#             [2, 10],
#             [5, 8],
#             [1, 2]
#         ]

#         for i in range(100):

#             clusters = [[] for i in range(self.k)]
#             for v in data:
#                 distance = []
#                 for c in self.centroid:
#                     dis = math.sqrt(
#                         sum((v[i] - c[i])** 2 for i in range(len(v)))
#                     )
#                     distance.append(dis)

#                 cluster_index = distance.index(min(distance))
#                 clusters[cluster_index].append(v)
#             print("Clusters:", clusters)

#             nc = []
#             for cluster in clusters:
#                 if len(cluster) > 0:
#                     centroid = []
#                     for i in range(len(cluster[0])):
#                         mean = sum(v[i] for v in cluster) / len(cluster)
#                         centroid.append(mean)
#                     nc.append(centroid)
#                 else:
#                     nc.append([0] * len(self.centroid[0]))

#             print("New Centroids:", nc)

#             if nc == self.centroid:
#                 break
#             self.centroid = nc
#         self.clusters = clusters


#     def show(self):
#         print("Clusters:")
#         print(self.clusters)
#         print("Centroid:")
#         print(self.centroid)

#     def predict(self, value):
#         distance = []
#         for centroid in self.centroid:
#             dis = math.sqrt(sum((value[i] - centroid[i])**2 for i in range(len(value))))
#             distance.append(dis)
#         return distance.index(min(distance))


# # -------------------------------------------------

# model = Kmean(3)
# data = [[2, 10],[2, 5],[8, 4],[5, 8],[7, 5],[6, 4],[1, 2],[4, 9]]
# model.fit(data)
# model.show()


# value = [3, 7]
# p = model.predict(value)
# print(value, "Belongs to Cluster", model.clusters[p])