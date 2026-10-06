import numpy as np
import matplotlib.pyplot as plt


class Kmean:

    def __init__(self, k):
        self.k = k
        self.centroid = None
        self.clusters = None

    # -------------------------------------------------
    # K-MEANS FIT
    # -------------------------------------------------

    def fit(self, data):

        self.centroid = [4, 11]

        for i in range(100):

            clusters = [[] for i in range(self.k)]

            # Assign values to nearest centroid
            for v in data:

                distance = []

                for c in self.centroid:
                    dis = abs(v - c)
                    distance.append(dis)

                cluster_index = distance.index(min(distance))

                clusters[cluster_index].append(v)

            # New centroids
            nc = []

            for cluster in clusters:

                if len(cluster) > 0:
                    nc.append(sum(cluster) / len(cluster))
                else:
                    nc.append(0)

            # Stop if centroid does not change
            if nc == self.centroid:
                break

            self.centroid = nc

        self.clusters = clusters

    # -------------------------------------------------
    # SHOW RESULT
    # -------------------------------------------------

    def show(self):

        print("Clusters:")
        print(self.clusters)

        print()

        print("Centroid:")
        print(self.centroid)

        print()

    # -------------------------------------------------
    # PREDICT
    # -------------------------------------------------

    def predict(self, value):

        distance = []

        for centroid in self.centroid:

            dis = abs(value - centroid)

            distance.append(dis)

        cluster_index = distance.index(min(distance))

        print(
            value,
            "Belongs to Cluster",
            self.clusters[cluster_index]
        )

        return self.clusters[cluster_index]

    # =================================================
    # GRAPH USING NORMAL PYTHON
    # =================================================

    def showgraph(self):

        plt.figure(figsize=(10, 5))

        for i in range(self.k):

            cluster = self.clusters[i]

            # X-axis positions
            x = range(len(cluster))

            # Y-axis values
            y = cluster

            plt.scatter(
                x,
                y,
                label=f"Cluster {i}"
            )

        # Centroids
        for i in range(self.k):

            plt.axhline(
                y=self.centroid[i],
                linestyle="--",
                label=f"Centroid {i}"
            )

        plt.title("K-Means Graph")
        plt.xlabel("Data Point Index")
        plt.ylabel("Value")

        plt.legend()
        plt.grid()

        plt.show()







# =================================================
# MAIN
# =================================================

model = Kmean(2)

data = [2, 4, 10, 12, 3, 20, 30, 11, 25]

model.fit(data)

model.show()

value = 30

model.predict(value)


model.showgraph()
