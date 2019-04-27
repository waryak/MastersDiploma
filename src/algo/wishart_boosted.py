import numpy as np
from scipy.special import gamma
from scipy.spatial import cKDTree
from collections import namedtuple


def construct_sorted_vertexes_matrix(z_vectors, k_neighbors):
    """
    Finding k nearest neighbor's indexes and distances to them for each z-vector.
    Sorting z-vectors according to distances to their's k nearest neighbor in ascending order.
    So, z-vectors with smallest distances to their neighbors are on the top.

    :param z_vectors:   Z-vectors, reconstructed from the original time series
    :param kdtree:      KDtree, fitted on reconstructed z-vectors
    :param k_neighbors: K nearest neighbor to look the distance to

    :return: Numpy arrays with indexes of nearest neigbors and distances to them
    """
    ckdtree = cKDTree(z_vectors)
    zvector_neighbors_dist, zvector_neighbors_ind = ckdtree.query(z_vectors, k_neighbors + 1)
    sorted_vertexes = np.argsort(zvector_neighbors_dist[:, -1])
    return zvector_neighbors_dist[sorted_vertexes, 1:], zvector_neighbors_ind[sorted_vertexes, 1:], sorted_vertexes


def density_near_zvector(radius, k_neighbors, n_zvectors, dimension):
    """
    Calculates the volume of N-dimensional ball

    :param radius:      Radius of the ball
    :param k_neighbors: K nearest neighbors is the space dimension of the hyper-ball
    :param n_clusters:  Number of z-vectors in the space

    :return: Volume of the ball
    """
    volume = (radius ** dimension) * (np.pi ** (k_neighbors / 2)) / gamma(k_neighbors / 2 + 1)
    density = k_neighbors / (volume * n_zvectors)
    return density

Cluster = namedtuple('Cluster', ['min_volume', 'max_volume', 'size', 'cluster_type', 'cluster_label'])

class Wishart:
    """
    Implementation of basic Wishart algorithm.
    Using numpy and cKDTree from scipy.
    """

    def __init__(self, wishart_neighbors, significance_level, dimension):
        self.wishart_neighbors = wishart_neighbors  # Number of neighbors
        self.G = np.array([], dtype=int)  # "Graph" array which stores all nodes.
        self.significance_level = significance_level  # Significance level
        self.completed_clusters = set()  # Set of completed clusters
        self.significant_clusters = set()  # Set of significant clusters
        self.clusters = None  # List of cluster labels for z-vectors
        self.dimension = dimension
        self.ClusterTuple = Cluster
        self.noize_class = self.ClusterTuple(min_volume=np.inf,
                                             max_volume=0,
                                             size=0,
                                             cluster_type="Noize",
                                             cluster_label=0)
        self.cluster_tuples = [self.noize_class]

    def __getstate__(self):
        """
        This function defines the fields of the class to be pickled.
        :return: All class fields except for "pool". It cannot be serialized.
        """
        self_dict = self.__dict__.copy()
        return self_dict

    def _clusters_to_completed(self, cluster_labels):
        """

        :param cluster_labels:
        :return:
        """
        for cluster_label in cluster_labels:
            cluster = self.cluster_tuples[int(cluster_label)]
            cluster = cluster._replace(cluster_type="Completed")
            self.cluster_tuples[int(cluster_label)] = cluster

    def _clusters_to_noize(self, cluster_labels):
        """

        :param cluster_labels:
        :return:
        """
        noize_cluster = self.cluster_tuples[0]
        for cluster_label in sorted(cluster_labels, reverse=True):
            cluster = self.cluster_tuples[cluster_label]
            cluster_size = cluster.size
            noize_cluster = noize_cluster._replace(size=noize_cluster.size + cluster_size)
            self.cluster_tuples[cluster_label] = None
        self.cluster_tuples[0] = noize_cluster

    def _add_vertex_to_cluster(self, vertex_index, vertex_radius, cluster_label):
        """

        :param vertex_index:
        :param vertex_radius:
        :param cluster_label:
        :return:
        """
        cluster = self.cluster_tuples[cluster_label]
        cluster_type = cluster.cluster_type
        vertex_volume = density_near_zvector(radius=vertex_radius,
                                             k_neighbors=self.wishart_neighbors,
                                             n_zvectors=len(self.clusters),
                                             dimension=self.dimension)

        if vertex_volume < cluster.min_volume:
            cluster = cluster._replace(min_volume=vertex_volume)
            difference = abs(vertex_volume - cluster.max_volume)
            if cluster_type in ("Completed", "Significant", "Noize"):
                pass
            else:
                if (difference > self.significance_level) & (cluster_label != 0):
                    self.significant_clusters.add(cluster_label)
                    cluster = cluster._replace(cluster_type="Significant")

        elif vertex_volume > cluster.max_volume:
            cluster = cluster._replace(max_volume=vertex_volume)
            difference = abs(vertex_volume - cluster.min_volume)
            if cluster_type in ("Completed", "Significant", "Noize"):
                pass
            else:
                if (difference > self.significance_level) & (cluster_label != 0):
                    self.significant_clusters.add(cluster_label)
                    cluster = cluster._replace(cluster_type="Significant")

        cluster = cluster._replace(size=cluster.size + 1)
        self.cluster_tuples[cluster_label] = cluster

    def _merge_two_clusters(self, cluster_reciever_label, cluster_to_be_merged_label):
        """

        :param cluster_label_1:
        :param cluster_label_2:
        :return:
        """
        cluster_reciever = self.cluster_tuples[cluster_reciever_label]
        cluster_to_be_merged = self.cluster_tuples[cluster_to_be_merged_label]

        min_volume = min(cluster_reciever.min_volume, cluster_to_be_merged.min_volume)
        max_volume = max(cluster_reciever.max_volume, cluster_to_be_merged.max_volume)

        if abs(max_volume - min_volume) > self.significance_level:
            cluster_type = "Significant"
            self.significant_clusters.add(cluster_reciever_label)
        else:
            cluster_type = "Non-Significant"
        new_cluster = self.ClusterTuple(min_volume=min_volume,
                                        max_volume=max_volume,
                                        size=cluster_reciever.size + cluster_to_be_merged.size,
                                        cluster_type=cluster_type,
                                        cluster_label=cluster_reciever.cluster_label)
        self.cluster_tuples[cluster_reciever_label] = new_cluster
        self.cluster_tuples[cluster_to_be_merged_label] = None

    # TODO: FIX "vertices_radiuses"
    def _check_cluster_significance(self, cluster, matrix_distances) -> bool:
        """
        Checks the cluster significance. If the cluster is significant - it's added to corresponding set.

        :param cluster: number of cluster to check.
        :param matrix_distances: matrix of distances

        :return: Returns bool. True - if cluster is significant and needs to be added to the list of significant.
                               False - if cluster is not significant doesn't need to be added to the list.
        """

        assert cluster in self.clusters, "There is no presence of cluster %i in the graph yet" % cluster
        vertices_radiuses = matrix_distances[self.clusters == cluster, self.wishart_neighbors - 1]

        vertex_with_biggest_radius = max(vertices_radiuses)
        vertex_with_smallest_radius = min(vertices_radiuses)

        ball_volume_max = density_near_zvector(radius=vertex_with_biggest_radius,
                                               k_neighbors=self.wishart_neighbors,
                                               n_zvectors=len(self.clusters),
                                               dimension=self.dimension)
        ball_volume_min = density_near_zvector(radius=vertex_with_smallest_radius,
                                               k_neighbors=self.wishart_neighbors,
                                               n_zvectors=len(self.clusters),
                                               dimension=self.dimension)

        maximum_difference = abs(ball_volume_max - ball_volume_min)
        if maximum_difference > self.significance_level:
            self.significant_clusters.add(cluster)
            return True
        else:
            return False

    # TODO: Think of the oprimisation in "CONNECTION SEARCH"

    def _case_1(self, zvector_index, radius) -> int:
        """
        Case 1: if the incoming z-vector is isolated.
        Turn the incoming z-vector to a new cluster with a new label.

        :param zvector_index: index of z-vector, which i going to be assigned as a new cluster

        :return: Label of the cluster
        """

        max_cluster = len(self.cluster_tuples)

        self.clusters[zvector_index] = max_cluster
        vertex_volume = density_near_zvector(radius=radius,
                                             k_neighbors=self.wishart_neighbors,
                                             n_zvectors=len(self.clusters),
                                             dimension=self.dimension)
        new_cluster = self.ClusterTuple(min_volume=vertex_volume,
                                        max_volume=vertex_volume,
                                        size=1,
                                        cluster_type="isolated",
                                        cluster_label=max_cluster)
        self.cluster_tuples.append(new_cluster)

        return max_cluster

    def _case_2(self, unique_clusters, vertex: int, matrix_distances: np.ndarray, radius) -> int:
        """
        Case 2: if the incoming z-vector is connected to the only cluster.
        - If cluster is completed, then turn incoming z-vector to "0"(background) noise
        - If cluster is not completed, then assign z-vector to this cluster

        :return:
        """
        # Get label of that single class
        connected_cluster: int = int(list(unique_clusters)[0])
        # If cluster is already completed, turn z-vector into "0" cluster
        if connected_cluster in self.completed_clusters:
            self.clusters[vertex] = 0
            return 0
        # If cluster is not completed
        else:
            self._add_vertex_to_cluster(vertex_index=vertex,
                                        vertex_radius=radius,
                                        cluster_label=connected_cluster)
            self.clusters[vertex] = connected_cluster
            # Check for significance
            return connected_cluster

    def _case_3(self, unique_clusters, vertex: int, matrix_distances: np.ndarray, radius) -> int:
        """
        Case 3: If incoming z-vector is connected to L={l1, l2, ...} different clusters and len(L) > 1.
        - If all clusters from L are completed, assign incoming z-vector to "0" cluster.
        - If there are >1 significant clusters in L or min(L) is 0, then:
            - Assign incoming cluster to "0".
            - Turn significant clusters to completed.
            - Turn all not-significant clusters to "0".
        - If there are <=1 significant clusters in L and min(L) is >0, then:
            - Collapse clusters l2, l3 into cluster l1, which is min(L)
            - Assign incoming z-vector with l1's label

        :return:
        """
        # If all clusters are completed:
        if all(map(lambda x: x in self.completed_clusters, unique_clusters)):
            self.clusters[vertex] = 0
            return 0
        elif min(unique_clusters) == 0 | len(unique_clusters & self.significant_clusters) > 1:
            self.clusters[vertex] = 0
            # Insignificant clusters turned into "background" clusters
            insignificant_to_zero = list(unique_clusters - self.significant_clusters - self.completed_clusters)
            if len(insignificant_to_zero) > 0:
                self.clusters[np.isin(self.cluster_tuples, np.array(insignificant_to_zero))] = 0
                self._clusters_to_noize(insignificant_to_zero)
            significant_to_completed = unique_clusters & self.significant_clusters
            self.completed_clusters = self.completed_clusters | significant_to_completed
            # Clusters, which became completed are not significant any more, so exclude them
            self.significant_clusters = self.significant_clusters - significant_to_completed
            self._clusters_to_completed(list(significant_to_completed))
            return 0
        else:
            # TODO: Ask if we can collapse into *completed* cluster or not (status quo: completed clusters are all thrown out)
            oldest_cluster = min(unique_clusters)
            self.clusters[vertex] = oldest_cluster
            self._add_vertex_to_cluster(vertex_index=vertex,
                                        vertex_radius=radius,
                                        cluster_label=int(oldest_cluster))
            clusters_to_be_merged = list(unique_clusters - {oldest_cluster} - self.completed_clusters)
            for cluster in sorted(clusters_to_be_merged, reverse=True):
                self.significant_clusters = self.significant_clusters - {cluster} - self.completed_clusters
                self.clusters[self.clusters == cluster] = oldest_cluster
                self._merge_two_clusters(cluster_reciever_label=int(oldest_cluster),
                                         cluster_to_be_merged_label=int(cluster)),
            return oldest_cluster

    def _find_connections(self, vertex, vertex_neighbors, matrix_indecies):
        """
        Finds all connection of vertex in a graph.

        :param vertex: vertex(z-vector) we want to find connections for.
        :param vertex_neighbors: a list of vertex neighbors.
        :param matrix_indexes: a matrix of all neighbors indexes for each vertex

        :return: vertex_to_g_connections - array of vertex indexes which are connected to the current vertex
                 vertex_to_g_connections_clusters - array of cluster labels of vertexes in vertex_to_g_connections
        """
        # Check if existing nodes are connected to a new-comming node
        node_in_g_connections = matrix_indecies[:self.G.shape[0]]
        vertex_to_g_connections = np.any(a=(node_in_g_connections == vertex), axis=1)
        if vertex_to_g_connections.shape[0] > 0:
            vertex_to_g_connections = self.G[vertex_to_g_connections]
        # Check if new-coming node has neighbors among existing nodes
        for neighbor in vertex_neighbors:
            if neighbor in self.G:
                vertex_to_g_connections = np.append(arr=vertex_to_g_connections, values=neighbor)
        vertex_to_g_connections_clusters = self.clusters[vertex_to_g_connections]
        return vertex_to_g_connections, vertex_to_g_connections_clusters

    def _create_graph(self, m_d, m_i, v_s) -> list:
        """
        Runs through all z-vectors and adds them to a graph, including them to some cluster or creating a new one

        :return: A list with cluster labels, which were given to z-vectors in order of iteration
        """
        cluster_labels = []

        for vertex, vertex_neighbors, neighbor_distances in zip(v_s, m_i, m_d):
            _, vertex_connections_clusters = self._find_connections(vertex=vertex,
                                                                    vertex_neighbors=vertex_neighbors,
                                                                    matrix_indecies=m_i)
            # Do not consider completed clusters in conditions.
            unique_clusters = set(vertex_connections_clusters)

            if len(unique_clusters) == 0:
                cluster_label = self._case_1(zvector_index=vertex,
                                             radius=neighbor_distances[-1])

            elif len(unique_clusters) == 1:
                cluster_label = self._case_2(unique_clusters=unique_clusters,
                                             vertex=vertex,
                                             matrix_distances=m_d,
                                             radius=neighbor_distances[-1])
            else:
                cluster_label = self._case_3(unique_clusters=unique_clusters,
                                             vertex=vertex,
                                             matrix_distances=m_d,
                                             radius=neighbor_distances[-1])
            cluster_labels.append(cluster_label)
            self.G = np.append(arr=self.G, values=vertex)
        return cluster_labels

    def _compute_completed_cluster_centers(self, z_vectors, sorted_vertex_indexes) -> None:
        """
        Computes cluster centers of all completed clusters and saves them to dict with scheme:
        {cluster label: {"cluster center": list[float], "cluster size": int}}

        :param sorted_vertex_indexes: Sorted indexes of the Initial z-vectors array
        :param z_vectors:             Initial z-vectors (not sorted)
        """
        completed_clusters_centers = {}
        # Resort initial z-vectors to the proper order
        z_vectors = z_vectors[sorted_vertex_indexes]
        for cluster in self.completed_clusters:
            cluster_mask = (self.clusters == cluster)
            cluster_zvectors = z_vectors[cluster_mask]
            cluster_center = cluster_zvectors.mean(axis=0)
            cluster_dict = {"center": cluster_center.tolist(),
                            "size": cluster_mask.sum()}
            completed_clusters_centers[cluster] = cluster_dict
        self.completed_clusters_centers = completed_clusters_centers

    def run_wishart(self, z_vectors: np.ndarray) -> list:
        """
        Runs all steps in Wishart algorithm.

        :param z_vectors: Initial array with z-vectors
        :return:          List of cluster labels in the iteration order.
        """

        print("--> Initializing clusters for %i vertexes" % z_vectors.shape[0])

        self.clusters = np.zeros(z_vectors.shape[0]) - 1
        print("--> Constructing vertex data")
        md_sorted, mi_sorted, zv_sorted = construct_sorted_vertexes_matrix(z_vectors,
                                                                           k_neighbors=self.wishart_neighbors)
        print("--> Fitting the graph")
        list_with_labels = self._create_graph(m_d=md_sorted, m_i=mi_sorted, v_s=zv_sorted)
        print("--> Finding the cluster centers")
        self._compute_completed_cluster_centers(z_vectors=z_vectors,
                                                sorted_vertex_indexes=zv_sorted)
        return list_with_labels


