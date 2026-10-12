import numpy as np


class Kmean:

    def __init__(self, k):
        self.k = k
        self.centroid = None
        self.clusters = None

    # -------------------------------------------------
    # FIT
    # -------------------------------------------------

    def fit(self, data):

        data = np.array(data, dtype=float)

        # Initial centroids
        self.centroid = np.array([
            [2, 10],
            [5, 8],
            [1, 2]
        ], dtype=float)

        for i in range(100):

            # -----------------------------------------
            # Distance
            # -----------------------------------------

            distances = np.sqrt(
                np.sum(
                    (data[:, None] - self.centroid) ** 2,
                    axis=2
                )
            )

            # -----------------------------------------
            # Nearest Cluster
            # -----------------------------------------

            new_labels = np.argmin(distances, axis=1)

            # -----------------------------------------
            # Create Clusters
            # -----------------------------------------

            clusters = [
                data[new_labels == i]
                for i in range(self.k)
            ]

            print("Clusters:", clusters)

            # -----------------------------------------
            # New Centroids
            # -----------------------------------------

            nc = []

            for cluster in clusters:

                if len(cluster) > 0:
                    new_centroid = np.mean(cluster, axis=0)
                    nc.append(new_centroid)

                else:
                    nc.append(np.zeros(data.shape[1]))

            nc = np.array(nc)

            print("New Centroids:", nc)

            # -----------------------------------------
            # Stop when centroids are stable
            # -----------------------------------------

            if np.allclose(self.centroid, nc):
                break

            self.centroid = nc

        self.clusters = clusters

    # -------------------------------------------------
    # SHOW
    # -------------------------------------------------

    def show(self):

        print("\nClusters:")

        for i, cluster in enumerate(self.clusters):
            print(f"Cluster {i}:", cluster)

        print("\nCentroid:")
        print(self.centroid)

    # -------------------------------------------------
    # PREDICT
    # -------------------------------------------------

    def predict(self, value):

        value = np.array(value)

        # Distance from every centroid
        distance = np.sqrt(
            np.sum(
                (value - self.centroid) ** 2,
                axis=1
            )
        )

        # Nearest centroid
        cluster_index = np.argmin(distance)

        print(
            value,
            "Belongs to Cluster",
            cluster_index
        )

        print(
            "Cluster Data:",
            self.clusters[cluster_index]
        )

        return self.clusters[cluster_index]


# =================================================
# MAIN
# =================================================

model = Kmean(3)

data = [
    [2, 10],
    [2, 5],
    [8, 4],
    [5, 8],
    [7, 5],
    [6, 4],
    [1, 2],
    [4, 9]
]

model.fit(data)

model.show()

value = [3, 7]

model.predict(value)