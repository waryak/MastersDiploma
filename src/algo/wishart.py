import time
import numpy as np
from pickle import dump
from scipy.special import gamma
from multiprocessing import Pool
from scipy.spatial import cKDTree
from pathlib import PurePosixPath


# TODO: WISHART NEEDS HELP!!!!

# noinspection PyPackageRequirements
class Wishart:
    """
    Implementation of basic Wishart algorithm.
    Using numpy and cKDTree from scipy
    """

    def __init__(self, wishart_neighbors, significance_level):
        self.wishart_neighbors = wishart_neighbors  # Number of neighbors
        self.G = np.array([], dtype=int)  # "Graph" array which stores all nodes.
        self.significance_level = significance_level  # Significance level
        self.clusters_completenes = set()  # Set of completed clusters
        self.significant_clusters = set()  # Set of significant clusters

    def __getstate__(self):
        """
        This function defines the fields of the class to be pickled.
        :return: All class fields except for "pool". It cannot be serialized.
        """
        self_dict = self.__dict__.copy()
        return self_dict

    def _fit_kd_tree(self, z_vectors):
        """
        Fit cKDTree for z-vectors
        :param z_vectors: z-vectors ndarray
        :return: cKDTree
        """
        kdt = cKDTree(data=z_vectors)
        return kdt

    def _n_dim_ball_volume(self, radius):
        """
        Calculates the volume of the ball around some point.
        :param radius: the distance from initial z-vector to the furthest neighbor z-vector.
        :return: volume of the ball.
        """
        if hasattr(self, 'clusters'):
            volume = (radius ** self.wishart_neighbors) * (np.pi ** (self.wishart_neighbors / 2)) / gamma(self.wishart_neighbors / 2 + 1)
        else:
            raise Exception("Array for clusters storage was not created till now. Something is completly wrong")
        return self.wishart_neighbors / self.clusters.shape[0] / volume

    def _check_cluster_significance(self, cluster, matrix_distances):
        """
        Checks the cluster significance.
        :param cluster: number of cluster to check.
        :param matrix_distances: matrix of distances
        :return: Returns bool. True - if cluster is significant and needs to be added to the list of significant.
                               False - if cluster is not significant doesn't need to be added to the list.
        """
        assert cluster in self.clusters, "There is no presence of cluster %i in the graph yet" % cluster
        verticies_radiuses = matrix_distances[self.clusters == cluster][:, self.wishart_neighbors - 1]
        vertex_with_biggest_radius = max(verticies_radiuses)
        vertex_with_smalles_radius = min(verticies_radiuses)
        ball_volume_max = self._n_dim_ball_volume(vertex_with_biggest_radius)
        ball_volume_min = self._n_dim_ball_volume(vertex_with_smalles_radius)
        maximum_difference = abs(ball_volume_max - ball_volume_min)
        if maximum_difference > self.significance_level:
            return True
        else:
            return False

    def _construct_neighbors_matrix(self, z_vectors, kdtree):
        """
        Construct ndarray to store neighbor matrix
        :param z_vectors: Ndarray of z-vectors.
        :param kdtree: cKDTree, fitted on z-vectors.
        :return:
        """
        zvector_neighbors_distances = np.zeros(shape=(z_vectors.shape[0], self.wishart_neighbors))
        zvector_neighbors_indexes = np.empty(shape=(z_vectors.shape[0], self.wishart_neighbors))
        self.clusters = np.zeros(z_vectors.shape[0], dtype=int)
        for index, z_vector in enumerate(z_vectors):
            zvector_neighbors_dist, zvector_neighbors_ind = kdtree.query(x=z_vector, k=self.wishart_neighbors + 1)
            zvector_neighbors_distances[index] = zvector_neighbors_dist[1:]
            zvector_neighbors_indexes[index] = zvector_neighbors_ind[1:]
        # sort matrix by ascending of the distance to the furthest NN
        vertecies_sorted = zvector_neighbors_distances[:, -1].argsort()
        matrix_distances_sorted = zvector_neighbors_distances[vertecies_sorted]
        matrix_indexes_sorted = zvector_neighbors_indexes[vertecies_sorted]
        return matrix_distances_sorted, matrix_indexes_sorted, vertecies_sorted

    def _form_new_cluster(self, zvector_index):
        """
        Forms a new cluster out of a new vertex
        :param zvector_index: index of z-vector, which i going to be assigned as a new cluster
        :return: Number of cluster
        """
        max_cluster = self.clusters.max() + 1
        self.clusters[zvector_index] = max_cluster
        return max_cluster

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
                _ = self._form_new_cluster(zvector_index=vertex)
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
                # If one of the clusters is zero, or there are more than one significant clusters, then
                # 1. assign new vertex to zero and
                # 2. significant clusters -> completed clusters and
                # 3. delete insignificant clusters
                elif (min(unique_clusters) == 0) | \
                        (len(unique_clusters.intersection(self.significant_clusters)) > 1):
                    self.clusters[vertex] = 0
                    insignificant_to_zero = unique_clusters.difference(self.significant_clusters)
                    significant_to_completed = unique_clusters.intersection(self.significant_clusters)
                    self.clusters_completenes = self.clusters_completenes.union(significant_to_completed)
                    # Clusters, which became completed are not significant any more, so exclude them
                    self.significant_clusters = self.significant_clusters.difference(significant_to_completed)
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
                        # If we collapse a significant cluster, we should exclude it from the set
                        self.significant_clusters = self.significant_clusters.difference({cluster})
                        self.clusters[self.clusters == cluster] = oldest_cluster
                    self.clusters[vertex] = oldest_cluster
                    if self._check_cluster_significance(cluster=oldest_cluster, matrix_distances=m_d):
                        self.significant_clusters.add(oldest_cluster)
            self.G = np.append(arr=self.G, values=vertex)

    def _form_cluster_centers(self, data):
        """

        :param data:
        :param reconstruction_shape:
        :return:
        """
        # TODO: Cluster centers MAKE SORTED!!!!! Because we need to choose their index
        cluster_centers = np.zeros(shape=(len(self.clusters_completenes), data.shape[1]))
        for cluster_index, cluster in enumerate(self.clusters_completenes):
            cluster_data = data[self.clusters == cluster]
            cluster_center = cluster_data.mean(axis=0)
            cluster_centers[cluster_index] = cluster_center
        self.cluster_centers = cluster_centers

    def _cluster_kdtree(self, n_lags):
        """
        Cuts cluster-centers and find k-nn for them.
        For examples if clusters centers are of size 7 and n_lags = 5, t
        hen we would be able to predict 2 points ahead.
        :param n_lags: number of elements from cluster center's z-vectors to use in prediction
        """
        lagged_data = self.cluster_centers[:, :n_lags]
        print("LAGGED DATA: ", lagged_data.shape)
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


class ParallelWishart:
    """
    Parallelisation of Wishart algorithm
    """

    def __init__(self, MODEL_PATH: PurePosixPath, k: int, h: float, n_processes: int = 2):
        self.k = k
        self.h = h
        self.n_processes = n_processes
        self.pool = Pool(processes=n_processes)
        self.MODEL_PATH = MODEL_PATH

    def __getstate__(self):
        """
        This function defines the fields of the class to be pickled.
        :return: All class fields except for "pool". It cannot be serialized.
        """
        self_dict = self.__dict__.copy()
        del self_dict["pool"]
        return self_dict

    def _run_single_wishart(self, args):
        """

        :param args:
        :return: path to the saved model
        """
        ws = Wishart(wishart_neighbors=self.k, significance_level=self.h)
        print("--> Fitting KDTree")
        kdt = ws._fit_kd_tree(z_vectors=args["z_vectors"])
        print("--> Constructing vertex data")
        m_d, m_i, v_s = ws._construct_neighbors_matrix(z_vectors=args["z_vectors"], kdtree=kdt)
        m_i = m_i.astype(int)
        print("--> Fitting the graph")
        ws._form_graph(m_d=m_d, m_i=m_i, v_s=v_s)
        print("--> Finding the cluster centers")
        ws._form_cluster_centers(data=m_d)
        print("--> Fitting clusters KDTree")
        ws._cluster_kdtree(n_lags=args["n_lags"])
        path = self.MODEL_PATH / args["model_name"]
        with open(path, "wb") as f:
            dump(ws, f)
        return path

    def run_wishart(self, list_of_args):
        """
        Function, running a pool of wishart processes
        :param list_of_args: a list of argument dictionaries
        :return:
        """
        assert self.n_processes == len(list_of_args), "Number of processes is %d, but number of arguments is %d" % (
        self.n_processes, len(list_of_args))

        if self.n_processes == 1:
            print("Running single process mode")
            path = self._run_single_wishart(list_of_args[0])
            return path
        else:
            print("Running on %d processors" % self.n_processes)
            paths = self.pool.map(func=self._run_single_wishart, iterable=list_of_args)
            return paths


class ParallelWishart2(Wishart):
    """
    Parallelisation of Wishart algorithm
    """

    def __init__(self, MODEL_PATH: PurePosixPath, wishart_neighbors: int, significance_level: float, n_processes: int = 2):
        self.k = wishart_neighbors
        self.h = significance_level
        self.n_processes = n_processes
        self.pool = Pool(processes=n_processes)

    def fun(self):
        r = super.__init__()
        return r

    def _run_single_wishart(self, **args):
        """

        :param args:
        :return: path to the saved model
        """
        ws = Wishart(wishart_neighbors=self.k, significance_level=self.h)
        kdt = ws._fit_kd_tree(z_vectors=args["z_vectors"])
        m_d, m_i, v_s = ws._construct_neighbors_matrix(z_vectors=args["z_vectors"], kdtree=kdt)
        m_i = m_i.astype(int)
        ws._form_graph(m_d=m_d, m_i=m_i, v_s=v_s)
        path = self.MODEL_PATH / args["model_name"]
        with open(path, "wb") as f:
            dump(ws, f)
        return path

    def run_wishart(self, list_of_args):
        """
        Function, running a pool of wishart processes
        :param list_of_args: a list of argument dictionaries
        :return:
        """
        assert self.n_processes == len(list_of_args), "Number of processes is %d, but number of arguments " \
                                                      "is %d"(self.n_processes, len(list_of_args))

        if self.n_processes == 1:
            path = self._run_single_wishart(list_of_args[0])
            return path
        else:
            paths = self.pool.map(func=self._run_single_wishart, iterable=list_of_args)
            return paths
