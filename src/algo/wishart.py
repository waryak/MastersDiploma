import pickle as pkl
from pathlib import PurePosixPath
from typing import Dict, List

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import gamma

from src.datamart.utils import parse_model_name


# TODO: WISHART NEEDS HELP!!!!

# noinspection PyPackageRequirements

# TODO: Think, i we should place KDTree in this function??
# TODO: CHeck, what is faster: 1) append to list in loop and then np.array() 2) allocate np array and assign to index in the loop
def construct_distance_matrix(z_vectors, kdtree, k_neighbors):
    """
    Finding k nearest neighbor's indexes and distances to them for each z-vector.

    :param z_vectors:   Z-vectors, reconstructed from the original time series
    :param kdtree:      KDtree, fitted on reconstructed z-vectors
    :param k_neighbors: K nearest neighbor to look the distance to

    :return: Numpy arrays with indexes of nearest neigbors and distances to them
    """
    # Prepare containers to store indexes of neighbors and distances to them
    zvector_neighbors_indexes = np.empty(shape=(z_vectors.shape[0], k_neighbors))
    zvector_neighbors_distances = np.zeros(shape=(z_vectors.shape[0], k_neighbors))
    # Finding nearest neighbors for each z-vector and distance to it
    # TODO: jit or map for this loop
    for index, z_vector in enumerate(z_vectors):
        zvector_neighbors_dist, zvector_neighbors_ind = kdtree.query(x=z_vector, k=k_neighbors + 1)
        zvector_neighbors_distances[index] = zvector_neighbors_dist[1:]
        zvector_neighbors_indexes[index] = zvector_neighbors_ind[1:]
    return zvector_neighbors_distances, zvector_neighbors_indexes.astype(int)


def sort_z_vectors_by_neighbors(zvector_neighbors_distances, zvector_neighbors_indexes):
    """

    Sorting z-vectors according to distances to their's k nearest neighbor in ascending order.
    So, z-vectors with smallest distances to their neighbors are on the top.

    :param zvector_neighbors_distances: Matrix with distances to k-th neighbor
    :param zvector_neighbors_indexes:   Matrix with indexes of neighbored z-vectors

    :return: Indexes of z-vectors sorted by the distance to k-th neighbor with
             distances and indexes matrices.
    """
    # Sorting by the k-th neighbor
    vertecies_sorted = zvector_neighbors_distances[:, -1].argsort()
    # Reselecting distances and indexes according to the sort
    matrix_distances_sorted = zvector_neighbors_distances[vertecies_sorted]
    matrix_indexes_sorted = zvector_neighbors_indexes[vertecies_sorted]
    # TODO: Make an asser here to check if the order is really ascending
    return vertecies_sorted, matrix_distances_sorted, matrix_indexes_sorted


def density_near_zvector(radius, k_neighbors, n_zvectors):
    """
    Calculates the volume of N-dimensional ball

    :param radius:      Radius of the ball
    :param k_neighbors: K nearest neighbors is the space dimension of the hyper-ball
    :param n_clusters:  Number of z-vectors in the space

    :return: Volume of the ball
    """
    volume = (radius ** k_neighbors) * (np.pi ** (k_neighbors / 2)) / gamma(k_neighbors / 2 + 1)
    density = k_neighbors / (volume * n_zvectors)
    return density


class Wishart:
    """
    Implementation of basic Wishart algorithm.
    Using numpy and cKDTree from scipy.
    """

    def __init__(self, wishart_neighbors, significance_level):
        self.wishart_neighbors = wishart_neighbors  # Number of neighbors
        self.G = np.array([], dtype=int)  # "Graph" array which stores all nodes.
        self.significance_level = significance_level  # Significance level
        self.completed_clusters = set()  # Set of completed clusters
        self.significant_clusters = set()  # Set of significant clusters
        self.clusters = None  # List of cluster labels for z-vectors

    def __getstate__(self):
        """
        This function defines the fields of the class to be pickled.
        :return: All class fields except for "pool". It cannot be serialized.
        """
        self_dict = self.__dict__.copy()
        return self_dict

    def _initialize_clusters(self, z_vectors):
        """
        Initialize z-vector's with clusters labels

        :param z_vectors: Z-vectors array

        :return: Number of clusters
        """
        self.clusters = np.zeros(z_vectors.shape[0])
        return len(self.completed_clusters)

    def _check_cluster_significance(self, cluster, matrix_distances) -> bool:
        """
        Checks the cluster significance. If the cluster is significant - it's added to corresponding set.

        :param cluster: number of cluster to check.
        :param matrix_distances: matrix of distances

        :return: Returns bool. True - if cluster is significant and needs to be added to the list of significant.
                               False - if cluster is not significant doesn't need to be added to the list.
        """

        assert cluster in self.clusters, "There is no presence of cluster %i in the graph yet" % cluster
        vertices_radiuses = matrix_distances[self.clusters == cluster][:, self.wishart_neighbors - 1]
        vertex_with_biggest_radius = max(vertices_radiuses)
        vertex_with_smallest_radius = min(vertices_radiuses)
        ball_volume_max = density_near_zvector(radius=vertex_with_biggest_radius,
                                               k_neighbors=self.wishart_neighbors,
                                               n_zvectors=len(self.clusters))
        ball_volume_min = density_near_zvector(radius=vertex_with_smallest_radius,
                                               k_neighbors=self.wishart_neighbors,
                                               n_zvectors=len(self.clusters))
        maximum_difference = abs(ball_volume_max - ball_volume_min)
        if maximum_difference > self.significance_level:
            self.significant_clusters.add(cluster)
            return True
        else:
            return False

    def _sort_zvectors(self, z_vectors: np.ndarray, kdtree: cKDTree):
        """

        :param z_vectors:
        :param kdtree:

        :return:
        """
        zv_n_distances, zv_n_indexes = construct_distance_matrix(z_vectors=z_vectors,
                                                                 kdtree=kdtree,
                                                                 k_neighbors=self.wishart_neighbors)
        zv_sorted, md_sorted, mi_sorted = sort_z_vectors_by_neighbors(zvector_neighbors_distances=zv_n_distances,
                                                                      zvector_neighbors_indexes=zv_n_indexes)
        return zv_sorted, md_sorted, mi_sorted

    # TODO: Think of the oprimisation in "CONNECTION SEARCH"

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

    def _case_1(self, zvector_index) -> int:
        """
        Case 1: if the incoming z-vector is isolated.
        Turn the incoming z-vector to a new cluster with a new label.

        :param zvector_index: index of z-vector, which i going to be assigned as a new cluster

        :return: Label of the cluster
        """
        max_cluster = self.clusters.max() + 1
        self.clusters[zvector_index] = max_cluster
        return max_cluster

    def _case_2(self, unique_clusters, vertex: int, matrix_distances: np.ndarray) -> int:
        """
        Case 2: if the incoming z-vector is connected to the only cluster.
        - If cluster is completed, then turn incoming z-vector to "0"(background) noise
        - If cluster is not completed, then assign z-vector to this cluster

        :return:
        """
        # Get label of that single class
        connected_cluster: int = list(unique_clusters)[0]
        # If cluster is already completed, turn z-vector into "0" cluster
        if connected_cluster in self.completed_clusters:
            self.clusters[vertex] = 0
            return 0
        # If cluster is not completed
        else:
            self.clusters[vertex] = connected_cluster
            # Check for significance
            _ = self._check_cluster_significance(cluster=connected_cluster, matrix_distances=matrix_distances)
            return connected_cluster

    def _case_3(self, unique_clusters, vertex: int, matrix_distances: np.ndarray) -> int:
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
        self.G = np.append(arr=self.G, values=vertex)
        # If all clusters are completed:
        if all(map(lambda x: x in self.completed_clusters, unique_clusters)):
            self.clusters[vertex] = 0
            return 0
        elif min(unique_clusters) == 0 | len(unique_clusters & self.significant_clusters) > 1:
            self.clusters[vertex] = 0
            # Insignificant clusters turned into "background" clusters
            insignificant_to_zero = unique_clusters - self.significant_clusters
            for insignificant_cluster in insignificant_to_zero:
                self.clusters[self.clusters == insignificant_cluster] = 0
            significant_to_completed = unique_clusters & self.significant_clusters
            self.completed_clusters = self.completed_clusters | significant_to_completed
            # Clusters, which became completed are not significant any more, so exclude them
            self.significant_clusters = self.significant_clusters - significant_to_completed
            return 0
        else:
            # TODO: Ask if we can collapse into *completed* cluster or not (status quo: completed clusters are all thrown out)
            oldest_cluster = min(unique_clusters)
            self.clusters[vertex] = oldest_cluster
            for cluster in (unique_clusters - {oldest_cluster}):
                self.significant_clusters = self.significant_clusters - {cluster}
                self.clusters[self.clusters == cluster] = oldest_cluster
            self._check_cluster_significance(cluster=oldest_cluster,
                                             matrix_distances=matrix_distances)
            return oldest_cluster

    def _create_graph(self, m_d, m_i, v_s) -> list:
        """
        Runs through all z-vectors and adds them to a graph, including them to some cluster or creating a new one

        :param m_d:
        :param m_i:
        :param v_s:

        :return: A list with cluster labels, which were given to z-vectors in order of iteration
        """
        cluster_labels = []
        for vertex, vertex_neighbors in zip(v_s, m_i):
            _, vertex_connections_clusters = self._find_connections(vertex=vertex,
                                                                    vertex_neighbors=vertex_neighbors,
                                                                    matrix_indecies=m_i)
            # Do not consider completed clusters in conditions.
            unique_clusters = set(vertex_connections_clusters) - self.completed_clusters

            if len(unique_clusters) == 0:
                cluster_label = self._case_1(zvector_index=vertex)
            elif len(unique_clusters) == 1:
                cluster_label = self._case_2(unique_clusters=unique_clusters,
                                             vertex=vertex,
                                             matrix_distances=m_d)
            else:
                cluster_label = self._case_3(unique_clusters=unique_clusters,
                                             vertex=vertex,
                                             matrix_distances=m_d)
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
        print("--> Initializing clusters for %i vertexes" %z_vectors.shape[0])
        self._initialize_clusters(z_vectors=z_vectors)
        print("--> Fitting KDTree")
        kdt = cKDTree(data=z_vectors)
        print("--> Constructing vertex data")
        zv_sorted, md_sorted, mi_sorted = self._sort_zvectors(z_vectors=z_vectors, kdtree=kdt)
        print("--> Fitting the graph")
        list_with_labels = self._create_graph(m_d=md_sorted, m_i=mi_sorted, v_s=zv_sorted)
        print("--> Finding the cluster centers")
        self._compute_completed_cluster_centers(z_vectors=z_vectors,
                                                sorted_vertex_indexes=zv_sorted)
        return list_with_labels

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

############################################################
# Most optimal structure is: Named tuple ("cluster_center", "cluster_size")
############################################################

# TODO: Play with cKDTree parameters
# TODO: Where to

def fit_kdtrees(path_to_models: List[PurePosixPath]) -> Dict[str, cKDTree]:
    """
    Iterates through model files and build cKDTree from cluster centers from the model

    :param path_to_models:

    :return:
    """
    models = {}
    predictions = {}
    for path_to_model in path_to_models:
        with open(path_to_model.as_posix(), "rb") as f:
            model = pkl.load(f)
        model_clusters_truncated = model.cluster_centers[:, :-1]
        # This is not the literal prediction. These last values can be used in final prediction
        prediction = model.cluster_centers[:, -1]
        kdtree = cKDTree(data=model_clusters_truncated)

        template = parse_model_name(path_to_model.name)
        models[str(template)] = kdtree
        predictions[str(template)] = prediction
    return models, predictions

def query_template_from_ts(ts: np.ndarray, template: List ) -> np.ndarray:
    """
    Queries a template sample from the time_series to make prediction with corresponding KDTree
    For example, if template is [2,2,3,2] and time series is [...,0,1,2,3,4,5,6,7,8,9], and
    we have to predict the next point: [...,0,1,2,3,4,5,6,7,8,9,x_to_predict],
    then the queried result would be [1, 3, 5, 8]

    :param ts:
    :param template:

    :return:
    """
    # Turning a template into indexing array
    template_reverse_indexing = np.cumsum(template[::-1])
    return ts[-template_reverse_indexing]

def template_prediction_mean(predictions_from_clusters: np.ndarray) -> float:
    return predictions_from_clusters.mean()
def aggregated_prediction_mean(predictions_from_templates: list) -> float:
    return np.mean(predictions_from_templates)


def predict_one_point_forward(dict_with_kdtrees: Dict[str, cKDTree],
                              dict_with_cluster_predictions: Dict[str, np.ndarray],
                              templates: List[List],
                              time_series: np.ndarray):
    """

    :param dict_with_kdtrees:
    :param dict_with_cluster_predictions:
    :param templates:
    :param time_series:
    :return:
    """
    # Here we will put predictions from templates (1 per each)
    predictions = []
    for template in templates:
        zvector_to_predict = query_template_from_ts(ts=time_series, template=template)
        template_cluster_predictions = dict_with_cluster_predictions[str(template)]
        template_kdtree = dict_with_kdtrees[str(template)]

        distances_to_clusters, cluster_indexes = template_kdtree.query(x=zvector_to_predict,
                                                           k=100,
                                                           distance_upper_bound=5)
        cluster_indexes = cluster_indexes[~np.isinf(distances_to_clusters)]
        neighbor_clusters_predictions = template_cluster_predictions[cluster_indexes]
        template_prediction = template_prediction_mean(neighbor_clusters_predictions)
        predictions.append(template_prediction)
    aggregated_prediction = aggregated_prediction_mean(predictions)
    return aggregated_prediction






def one_point_prediction(dict_with_kdtrees: Dict[str, cKDTree],
                         templates: List[List],
                         time_series: np.ndarray):
    """
    Make
    :return:
    """
    for template in templates:
        query_template_from_ts(ts=time_series, template=template)
        template_kdtree = dict_with_kdtrees[str(template)]





############################################################
# Most optimal structure is: Named tuple ("cluster_center", "cluster_size")
############################################################



# class WishartAggregator:
#     """
#     Class which works prepares distributed single-template wishart models in a ready-to-work with form
#     """
#     def __init__(self, path_to_models):
#         self.PATH_MODELS = path_to_models
#
#     def _list_all_models_files(self):
#         """
#         Lists all models files in given directory
#         :return: number of models files found
#         """
#         model_files = listdir(self.PATH_MODELS)
#         # Leave only files, not directories
#         model_files = [PurePosixPath(self.PATH_MODELS) / f for f in model_files if path.isfile(f)]
#         self.models_files = model_files
#         return len(self.models_files)
#
#     # TODO: Redefine the function so it works with the right format of
#     def _fit(self):
#         """
#
#         :return:
#         """
#         if hasattr(self, "models_files"):
#             print("Found model files in quantity of %i" % len(self.models_files))
#         else:
#             raise Exception("First initialize model files")
#
#         for model

