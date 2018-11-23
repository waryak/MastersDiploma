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
        """
        Fit KDTree for z-vectors
        :param data:
        :return:
        """
        kdt = cKDTree(data=data)
        return kdt

    def _n_dim_ball_volume(self, radius):
        """
        Calculates the volume of the ball
        :param radius: the distance to the furthest neighbor
        :return:
        """

        volume = (radius ** self.k) * (np.pi ** (self.k / 2)) / gamma(self.k / 2 + 1)
        return self.k / self.clusters.shape[0] / volume

    def _check_cluster_significance(self, cluster, matrix_distances):
        """
        Checks the cluster significance
        :param cluster: number of cluster to check
        :param matrix_distances: matrix of distances
        :return: Returns bool. True - if cluster is significant and was added to the list.
                               False - if cluster is not significant and was not added to the list.
        """
        assert cluster in self.clusters, "There is no presence of cluster %i in the graph yet" %cluster
        verticies_radiuses = matrix_distances[self.clusters == cluster][:, self.k - 1]
        vertex_with_biggest_radius = max(verticies_radiuses)
        vertex_with_smalles_radius = min(verticies_radiuses)
        ball_volume_max = self._n_dim_ball_volume(vertex_with_biggest_radius)
        ball_volume_min = self._n_dim_ball_volume(vertex_with_smalles_radius)
        maximum_difference = abs(ball_volume_max - ball_volume_min)
        if maximum_difference > self.h:
            return True
        else:
            return False

    def _construct_neighbors_matrix(self, data, kdtree):
        """

        :param data:
        :param kdtree:
        :return:
        """
        zvector_neighbors_distances = np.zeros(shape=(data.shape[0], self.k))
        zvector_neighbors_indexes = np.empty(shape=(data.shape[0], self.k))
        self.clusters = np.zeros(data.shape[0], dtype=int)
        for index, zvector in enumerate(data):
            zvector_neighbors_dist, zvector_neighbors_ind = kdtree.query(x=zvector, k=self.k + 1)
            zvector_neighbors_distances[index] = zvector_neighbors_dist[1:]
            zvector_neighbors_indexes[index] = zvector_neighbors_ind[1:]
        # sort matrix by ascending of the distance to the furthest NN
        vertecies_sorted = zvector_neighbors_distances[:, -1].argsort()
        matrix_distances_sorted = zvector_neighbors_distances[vertecies_sorted]
        matrix_indexes_sorted = zvector_neighbors_indexes[vertecies_sorted]
        return matrix_distances_sorted, matrix_indexes_sorted, vertecies_sorted

    def _form_new_cluster(self, vertex):
        """
        Forms a new cluster out of a new vertex
        :param vertex:
        :return: number of cluster
        """
        max_cluster = self.clusters.max() + 1
        self.clusters[vertex] = max_cluster
        return max_cluster

    def _find_connections(self, vertex, vertex_neighbors, matrix_indecies):
        """
        Finds all connection of this vertex in a graph.
        :param vertex: vertex we want to find connections for.
        :param vertex_neighbors: a list of vertex neighbors.
        :param matrix_indexes: a matrix of all neighbors indexes for each vertex
        :return: vertex_to_g_connections - array of vertex indexes which are connected to the current vertex
                 vertex_to_g_connections_clusters - array of cluster labels of vertexes in vertex_to_g_connections
        """
        # Check if existing nodes are connected to a new-comming node
        node_in_g_connections = matrix_indecies[self.G]
        vertex_to_g_connections = np.any(a=(node_in_g_connections == vertex), axis=1)
        if vertex_to_g_connections.shape[0] > 0:
            vertex_to_g_connections = self.G[vertex_to_g_connections]
        # Check if new-coming node has neighbors among existing nodes
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
                _ = self._form_new_cluster(vertex=vertex)
            # If vertex has only connection to one cluster, then:
            elif len(unique_clusters) == 1:
                connection_cluster = list(unique_clusters)[0]
                # If cluster is already completed
                if connection_cluster in self.clusters_completenes:
                    self.clusters[vertex] = 0
                # If cluster is not completed
                else:
                    self.clusters[vertex] = connection_cluster
                    if self._check_cluster_significance(connection_cluster, m_d):
                        # Add cluster to list of significant clusters
                        self.significant_clusters.add(connection_cluster)
            # If vertex is connected to more than one clusters/vertcies
            else:
                # If all connections are completed cluster, than assign vertex to zero
                if all(map(lambda x: x in self.clusters_completenes, unique_clusters)):
                    self.clusters[vertex] = 0
                # If one of the clusters is zero, or there are more than one significant clusters,
                # then assign new vertex to zero
                elif (min(unique_clusters) == 0) | \
                     (len(unique_clusters.intersection(self.significant_clusters)) > 1):
                    self.clusters[vertex] = 0
                    insignificant_to_zero = unique_clusters.difference(self.significant_clusters)
                    significant_to_completed = unique_clusters.intersection(self.significant_clusters)
                    self.clusters_completenes = self.clusters_completenes.union(significant_to_completed)
                    for cluster in insignificant_to_zero:
                        self.clusters[self.clusters == cluster] = 0
                # If there is one or less significant class and no zero classes,
                # then we should collapse all clusters including new-coming node
                # to the oldest cluster(oldest means that it has the biggest density)
                # TODO: Think of the oprimisation in "CONNECTION SEARCH"
                # TODO: DO NOT COLLAPSE ALL CLUSTERS. DO NOT TOUCH COMPLETED CLUSTERS!
                else:
                    oldest_cluster = min(unique_clusters)
                    # Exclude completed clusters
                    unique_clusters = unique_clusters.difference(self.clusters_completenes)
                    other_clusters = sorted(list(unique_clusters))[1:]
                    for cluster in other_clusters:
                        self.clusters[self.clusters == cluster] = oldest_cluster
                    self.clusters[vertex] = oldest_cluster
                    if self._check_cluster_significance(cluster=oldest_cluster, matrix_distances=m_d):
                        self.significant_clusters.add(oldest_cluster)
            self.G = np.append(arr=self.G, values=vertex)

    def _form_cluster_centers(self, data, reconstruction_shape):
        """

        :param data:
        :return:
        """
        # TODO: Cluster centers MAKE SORTED!!!!! Because we need to choose their index
        cluster_centers = np.zeros(shape=(len(self.clusters_completenes), reconstruction_shape))
        for cluster_index, cluster in enumerate(self.clusters_completenes):
            cluster_data = data[self.clusters == cluster]
            cluster_center = cluster_data.mean(axis=0)
            cluster_centers[cluster_index] = cluster_center
        self.cluster_centers = cluster_centers

    def _cluster_kdtree(self, n_lags):
        """
        """
        lagged_data = self.cluster_centers[:, :n_lags]
        centers_kdtree = cKDTree(data=lagged_data)
        self.centers_kdtree = centers_kdtree

    def predict(self, data, epsilon=1, max_neighbors=100):
        """
        """
        distances, clusters = self.centers_kdtree.query(x=data, k=max_neighbors, distance_upper_bound=epsilon)
        mask = ~np.isinf(distances)
        distances = distances[mask]
        clusters = clusters[mask]
        if (len(distances) > 0) & (len(clusters) > 0):
            distances = 1 / distances
            prediction = np.average(a=self.cluster_centers[clusters, -1], weights=distances)
        else:
            # Data was not recognised as something predictable
            prediction = np.nan
        return prediction






