import numpy as np
from scipy.special import gamma
from scipy.spatial import distance
from scipy.spatial import cKDTree

class Wishart:
    """
    Implementation of wishart algorithm
    """

    def __init__(self, k, h):
        self.k = k
        self.G = np.array([], dtype=int)
        self.h = h
        self.clusters_completenes = set()
        self.significant_clusters = set()

    def _fit_kd_tree(self, data):
        kdt = cKDTree(data=data)
        return kdt

    def _n_dim_ball_volume(self, radius):
        volume = (radius ** self.k) * (np.pi ** (self.k / 2)) / gamma(self.k / 2 + 1)
        return volume * self.k / self.clusters.shape[0]

    def _check_cluster_significance(self, cluster, matrix_distances):
        """
        """
        verticies_radiuses = matrix_distances[self.clusters == cluster][:, self.k - 1]
        vertex_with_biggest_radius = max(verticies_radiuses)
        vertex_with_smalles_radius = min(verticies_radiuses)
        ball_volume_max = self._n_dim_ball_volume(vertex_with_biggest_radius)
        ball_volume_min = self._n_dim_ball_volume(vertex_with_smalles_radius)
        maximum_difference = abs(ball_volume_max - ball_volume_min)
        if maximum_difference > self.h:
            self.significant_clusters.add(cluster)
        else:
            pass

    def _construct_neighbors_matrix(self, data, kdtree):
        zvector_neighbors_distances = np.zeros(shape=(data.shape[0], self.k))
        zvector_neighbors_indexes = np.empty(shape=(data.shape[0], self.k))
        self.clusters = np.zeros(data.shape[0], dtype=int)
        for index, zvector in enumerate(data):
            zvector_neighbors_dist, zvector_neighbors_ind = kdtree.query(x=zvector, k=self.k + 1)
            zvector_neighbors_distances[index] = zvector_neighbors_dist[1:]
            zvector_neighbors_indexes[index] = zvector_neighbors_ind[1:]
        # sort matricies
        vertecies_sorted = zvector_neighbors_distances[:, -1].argsort()
        matrix_distances_sorted = zvector_neighbors_distances[vertecies_sorted]
        matrix_indexes_sorted = zvector_neighbors_indexes[vertecies_sorted]
        return matrix_distances_sorted, matrix_indexes_sorted, vertecies_sorted

    def _form_new_cluster(self, vertex):
        max_cluster = self.clusters.max() + 1
        self.clusters[vertex] = max_cluster

    def _find_connections(self, vertex, vertex_neighbors, matrix_indecies):
        # Check if existing nodes are connected to a new-comming node
        node_in_g_connections = matrix_indecies[self.G]
        vertex_to_g_connections = np.any(a=(node_in_g_connections == vertex), axis=1)
        if vertex_to_g_connections.shape[0] > 0:
            vertex_to_g_connections = self.G[vertex_to_g_connections]
        # Check if new-comming node has neighbors among existing nodes
        for neighbor in vertex_neighbors:
            if neighbor in self.G:
                vertex_to_g_connections = np.append(arr=vertex_to_g_connections, values=neighbor)
        vertex_to_g_connections_clusters = self.clusters[vertex_to_g_connections]
        return vertex_to_g_connections, vertex_to_g_connections_clusters

    def _form_graph(self, m_d, m_i, v_s):
        """
        """
        for vertex, vertex_neighbors in zip(v_s, m_i):
            _, vertex_connections_clusters = self._find_connections(vertex=vertex,
                                                                    vertex_neighbors=vertex_neighbors,
                                                                    matrix_indecies=m_i)
            unique_clusters = set(vertex_connections_clusters)

            # Check if vertex is isolated
            if len(unique_clusters) == 0:
                self._form_new_cluster(vertex=vertex)
            # If vertex has only connection to one cluster, then:
            elif len(vertex_connections) == 1:
                vertex_connection = vertex_connections[0]
                # If cluster is already completed
                if vertex_connection in self.clusters_completenes:
                    self.clusters[vertex] = 0
                # If cluster is not completed
                else:
                    connection_cluster = self.clusters[vertex_connection]
                    self.clusters[vertex] = connection_cluster
                    self._check_cluster_significance(connection_cluster, m_d)
            # If vertex is connected to more than one clusters/vertcies
            else:
                connected_clusters = set(self.clusters[vertex_connections])
                # If all connections are completed cluster, than assign vertex to zero
                if all(map(lambda x: x in self.clusters_completenes, connected_clusters)):
                    self.clusters[vertex] = 0
                # If one of the clusters is zero, or there are more than one significant clusters,
                # then assign new vertex to zero
                elif (min(connected_clusters) == 0) | \
                        (len(connected_clusters.intersection(self.significant_clusters)) > 1):
                    self.clusters[vertex] = 0
                    insignificant_to_zero = connected_clusters.difference(self.significant_clusters)
                    significant_to_completed = connected_clusters.intersection(self.significant_clusters)
                    self.clusters_completenes = self.clusters_completenes.union(significant_to_completed)
                    for cluster in insignificant_to_zero:
                        self.clusters[self.clusters == cluster] = 0
                # If there is one or less significant class and no zero classes,
                # then we should collapse all clusters including new-comming node
                # to the oldest cluster(oldest means that it has the biggest density)
                else:
                    oldest_cluster = min(connected_clusters)
                    other_clusters = sorted(list(connected_clusters))[1:]
                    for cluster in other_clusters:
                        self.clusters[self.clusters == cluster] = oldest_cluster
                    self.clusters[vertex] = oldest_cluster
                    self._check_cluster_significance(cluster=oldest_cluster, matrix_distances=m_d)

            self.G = np.append(arr=self.G, values=vertex)

    def _form_cluster_centers(self, data):
        cluster_centers = np.zeros(shape=(len(self.clusters_completenes), self.k + 1))
        for cluster_index, cluster in enumerate(self.clusters_completenes):
            cluster_data = data[self.clusters == cluster]
            cluster_center = cluster_data.mean(axis=0)
            cluster_centers[cluster_index] = cluster_center
        return cluster_centers

    def _cluster_kdtree(self, n_lags, cluster_centers):
        """
        """
        lagged_data = cluster_centers[:, :n_lags]
        return cKDTree(data=lagged_data)

    def predict(self, data, cluster_kdtree, cluster_centers):
        """
        """
        prediction_indexes, prediction_distances = cluster_kdtree.query(x=data, k=self.k + 2)
        nearest_predictions_indexes = prediction_indexes[:, 1]
        return cluster_centers[nearest_predictions_indexes]


