import numpy as np


class Kmean:

    def __init__(self, k):
        self.k = k
        self.centroid = None
        self.clusters = None

    def fit(self, data):

        # Convert list into NumPy array
        data = np.array(data)

        # Initial centroids
        self.centroid = np.array([4, 11], dtype=float)

        for i in range(100):

            # Calculate distance of every value
            # from every centroid
            distances = np.abs(data[:, None] - self.centroid)

            # Find nearest centroid
            labels = np.argmin(distances, axis=1)

            # Create clusters
            clusters = [
                data[labels == i].tolist()
                for i in range(self.k)
            ]

            # Calculate new centroids
            new_centroid = np.array([
                cluster.mean() if len(cluster) > 0 else 0
                for cluster in clusters
            ])

            # Stop if centroid does not change
            if np.array_equal(new_centroid, self.centroid):
                break

            self.centroid = new_centroid

        self.clusters = clusters

    def show(self):

        print("Clusters:")
        print(self.clusters)

        print()

        print("Centroid:")
        print(self.centroid)

        print()

    def predict(self, value):

        # Calculate distance from every centroid
        distance = np.abs(value - self.centroid)

        # Find nearest centroid
        cluster_index = np.argmin(distance)

        print(
            value,
            "Belongs to Cluster",
            self.clusters[cluster_index]
        )

        return self.clusters[cluster_index]


# -------------------------------------------------

model = Kmean(2)

data = [2, 4, 10, 12, 3, 20, 30, 11, 25]

model.fit(data)

model.show()

value = 30

model.predict(value)