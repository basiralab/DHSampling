import math
import time
import scipy
# import metis
import torch
import random
import numpy as np
import networkx as nx
import skfuzzy as fuzz
from tqdm import tqdm
from IPython import embed
from scipy.cluster.vq import kmeans2
from scipy.special import kl_div
from sklearn.model_selection import train_test_split


import numpy as np
from sklearn import metrics


def mmd_linear(X, Y):
    """MMD using linear kernel (i.e., k(x,y) = <x,y>)
    Note that this is not the original linear MMD, only the reformulated and faster version.
    The original version is:
        def mmd_linear(X, Y):
            XX = np.dot(X, X.T)
            YY = np.dot(Y, Y.T)
            XY = np.dot(X, Y.T)
            return XX.mean() + YY.mean() - 2 * XY.mean()

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Returns:
        [scalar] -- [MMD value]
    """
    delta = X.mean(0) - Y.mean(0)
    return delta.dot(delta.T)


def mmd_rbf(X, Y, gamma=1.0):
    """MMD using rbf (gaussian) kernel (i.e., k(x,y) = exp(-gamma * ||x-y||^2 / 2))

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Keyword Arguments:
        gamma {float} -- [kernel parameter] (default: {1.0})

    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.rbf_kernel(X, X, gamma)
    YY = metrics.pairwise.rbf_kernel(Y, Y, gamma)
    XY = metrics.pairwise.rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def mmd_poly(X, Y, degree=2, gamma=1, coef0=0):
    """MMD using polynomial kernel (i.e., k(x,y) = (gamma <X, Y> + coef0)^degree)

    Arguments:
        X {[n_sample1, dim]} -- [X matrix]
        Y {[n_sample2, dim]} -- [Y matrix]

    Keyword Arguments:
        degree {int} -- [degree] (default: {2})
        gamma {int} -- [gamma] (default: {1})
        coef0 {int} -- [constant item] (default: {0})

    Returns:
        [scalar] -- [MMD value]
    """
    XX = metrics.pairwise.polynomial_kernel(X, X, degree, gamma, coef0)
    YY = metrics.pairwise.polynomial_kernel(Y, Y, degree, gamma, coef0)
    XY = metrics.pairwise.polynomial_kernel(X, Y, degree, gamma, coef0)
    return XX.mean() + YY.mean() - 2 * XY.mean()


def _k_means(data, node, num_center, minit='points'):
    '''
    TODO: functions to perform k-means on data
    Args:
        data: numpy array with size of NxM (N: #! of data; M: #! of feature size)
        num_center: label
        minit: initial type refer to (https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.kmeans2.html)
    Returns: centroid point with size of num_center x M, and label of each sample
    '''
    centroid, label = kmeans2(data, num_center, minit=minit)
    clusters = {}
    for item in range(num_center):
        index = np.where(label==item)
        node_tmp = [node[f] for f in index[0]]
        clusters[item] = node_tmp

    return clusters


def _fuzzy_k_means(data, node, num_center, er=1e-8):
    '''
    TODO: functions to perform fuzzy-k-means based on scikit-fuzzy
    Args:
        data: numpy array with size of NxM (N: #! of data; M: #! of feature size)
        num_center: member of each cluster
    Returns:
    '''
    data = data.T # MxN
    cntr, u_orig, _, _, _, _, _ = fuzz.cluster.cmeans(
        data, num_center, 2, error=0.001, maxiter=1000)
    threshold = 1/num_center+1e-5
    idx = u_orig.argsort(0)
    clusters_tmp_0 = idx[-1, :]
    clusters_tmp_1 = idx[-2, :]
    clusters_tmp_2 = idx[-3, :]

    cluster_membership = {}

    for idx in range(num_center):
        index_0 = np.where(clusters_tmp_0 == idx)
        index_0 = list(index_0[0])

        index_1 = np.where(clusters_tmp_1 == idx)
        index_1 = list(index_1[0])

        index_2 = np.where(clusters_tmp_2 == idx)
        index_2 = list(index_2[0])

        index = index_0
        index.extend(index_1)
        # index.extend(index_2)

        index = list(set(index))

        node_tmp = [node[f] for f in index]
        cluster_membership[idx] = node_tmp

    return cluster_membership


class ClusteringMachine(object):
    """
    Clustering the graph, feature set and target.
    """
    def __init__(self, args, graph, features, target):
        """
        :param args: Arguments object with parameters.
        :param graph: Networkx Graph.
        :param features: Feature matrix (ndarray).
        :param target: Target vector (ndarray).
        """
        self.args = args
        self.graph = graph
        self.features = features
        self.target = target
        self._set_sizes()
        self._train_test_split()

    def _set_sizes(self):
        """
        Setting the feature and class count.
        """
        self.feature_count = self.features.shape[1]
        self.class_count = np.max(self.target)+1
        self.nodes = [node for node in self.graph.nodes()]

    def _train_test_split(self, val_ratio=0.1, test_ratio=0.2):
        train_mask = np.zeros(len(self.nodes)) == 1
        val_mask = np.zeros(len(self.nodes)) == 1
        test_mask = np.zeros(len(self.nodes)) == 1
        val_split = int(len(self.nodes)*val_ratio)
        test_split = int(len(self.nodes)*(val_ratio+test_ratio))
        val_mask[:val_split] = True
        test_mask[val_split:test_split] = True
        train_mask[test_split:] = True
        self.train_nodes = {}
        self.val_nodes = {}
        self.test_nodes = {}

        for idx, node in enumerate(self.nodes):
            self.train_nodes[node] = train_mask[idx]
            self.test_nodes[node] = test_mask[idx]
            self.val_nodes[node] = val_mask[idx]

    def decompose(self):
        """
        Decomposing the graph, partitioning the features and target, creating Torch arrays.
        """
        if self.args.clustering_method == "metis":
            print("\nMetis graph clustering started.\n")
            self.metis_clustering()
        elif self.args.clustering_method == 'random_hyper':
            print("\nRandom hyper graph clustering started.\n")
            self.hyperedge_clustering_random()
        elif self.args.clustering_method == 'fuzzy_c_hyper':
            print("\nFuzzy_c_means hyper graph clustering started.\n")
            self.hyperedge_clustering_fuzzyc()
        elif self.args.clustering_method == 'knn':
            print("\nKNN graph clustering started.\n")
            self._knn_clustering()
        else:
            print("\nRandom graph clustering started.\n")
            self.random_clustering()
        self.general_data_partitioning()

        # self._diversity_matrix()
        self.transfer_edges_and_nodes()

    def random_clustering(self):
        """
        Random clustering the nodes.
        """
        self.clusters = [cluster for cluster in range(self.args.cluster_number)]
        clusters_tmp = np.array([random.choice(self.clusters) for idx in range(len(self.nodes))])
        self.cluster_membership = {}

        for idx in range(self.args.cluster_number):
            index = np.where(clusters_tmp==idx)
            index = list(index[0])
            node_tmp = [self.nodes[f] for f in index]
            self.cluster_membership[idx] = node_tmp

    def hyperedge_clustering_random(self):
        self.clusters = [cluster for cluster in range(self.args.cluster_number)]
        clusters_tmp_0 = np.array([random.choice(self.clusters) for idx in range(len(self.nodes))])
        clusters_tmp_1 = np.array([random.choice(self.clusters) for idx in range(int(len(self.nodes)))])
        clusters_tmp_2 = np.array([random.choice(self.clusters) for idx in range(len(self.nodes))])

        self.cluster_membership = {}

        for idx in range(self.args.cluster_number):
            index_0 = np.where(clusters_tmp_0==idx)
            index_0 = list(index_0[0])

            index_1 = np.where(clusters_tmp_1==idx)
            index_1 = list(index_1[0])

            index_2 = np.where(clusters_tmp_2==idx)
            index_2 = list(index_2[0])

            index = index_0
            index.extend(index_1)
            # index.extend(index_2)

            index = list(set(index))

            node_tmp = [self.nodes[f] for f in index]
            self.cluster_membership[idx] = node_tmp

    def hyperedge_clustering_fuzzyc(self):
        self.clusters = [cluster for cluster in range(self.args.cluster_number)]
        self.features_tmp = self.features[self.nodes]
        self.cluster_membership = _fuzzy_k_means(self.features_tmp, self.nodes, self.args.cluster_number)


    def _knn_clustering(self):
        self.clusters = [cluster for cluster in range(self.args.cluster_number)]
        self.features_tmp = self.features[self.nodes]
        self.cluster_membership = _k_means(self.features_tmp, self.nodes, self.args.cluster_number)

    def metis_clustering(self):
        """
        Clustering the graph with Metis. For details see:
        """
        (st, parts) = metis.part_graph(self.graph, self.args.cluster_number)
        self.clusters = list(set(parts))
        self.cluster_membership = {node: membership for node, membership in enumerate(parts)}

    def general_data_partitioning(self):
        """
        Creating data partitions and train-test splits.
        """
        self.sg_nodes = {}
        self.sg_edges = {}
        self.sg_train_nodes = {}
        self.sg_test_nodes = {}
        self.sg_val_nodes = {}
        self.sg_features = {}
        self.sg_targets = {}
        self.sg_graph = {}

        for cluster in self.clusters:
            subgraph = self.graph.subgraph([node for node in sorted(self.graph.nodes()) if node in self.cluster_membership[cluster]])
            self.sg_graph[cluster] = subgraph
            self.sg_nodes[cluster] = [node for node in sorted(subgraph.nodes())]
            self.sg_train_nodes[cluster] = [self.train_nodes[node] for node in self.sg_nodes[cluster]]
            self.sg_val_nodes[cluster] = [self.val_nodes[node] for node in self.sg_nodes[cluster]]
            self.sg_test_nodes[cluster] = [self.test_nodes[node] for node in self.sg_nodes[cluster]]

            mapper = {node: i for i, node in enumerate(sorted(self.sg_nodes[cluster]))}

            self.sg_edges[cluster] = [[mapper[edge[0]], mapper[edge[1]]] for edge in subgraph.edges()] +  [[mapper[edge[1]], mapper[edge[0]]] for edge in subgraph.edges()]
            self.sg_features[cluster] = self.features[self.sg_nodes[cluster],:]
            self.sg_targets[cluster] = self.target[self.sg_nodes[cluster],:]

    def transfer_edges_and_nodes(self):
        """
        Transfering the data to PyTorch format.
        """
        for cluster in self.clusters:
            self.sg_nodes[cluster] = torch.LongTensor(self.sg_nodes[cluster])
            self.sg_edges[cluster] = torch.LongTensor(self.sg_edges[cluster]).t()
            self.sg_train_nodes[cluster] = torch.LongTensor(self.sg_train_nodes[cluster])
            self.sg_test_nodes[cluster] = torch.LongTensor(self.sg_test_nodes[cluster])
            self.sg_val_nodes[cluster] = torch.LongTensor(self.sg_val_nodes[cluster])
            self.sg_features[cluster] = torch.FloatTensor(self.sg_features[cluster])
            self.sg_targets[cluster] = torch.LongTensor(self.sg_targets[cluster])

    def _select_k(self, spectrum, minimum_energy=0.9):
        running_total = 0.0
        total = sum(spectrum)
        if total == 0.0:
            return len(spectrum)
        for i in range(len(spectrum)):
            running_total += spectrum[i]
            if running_total / total >= minimum_energy:
                return i + 1
        return len(spectrum)

    def _feature_similarity(self, F_G1, F_G2):
        return mmd_rbf(F_G1, F_G2)

    def _similarity(self, G1, G2, Embed_G1, Embed_G2, lamba=0.5):
        # TODO: calculate Topology similarity
        laplacian1 = nx.spectrum.laplacian_spectrum(G1)
        laplacian2 = nx.spectrum.laplacian_spectrum(G2)

        k1 = self._select_k(laplacian1)
        k2 = self._select_k(laplacian2)

        k = min(k1, k2)

        Topology_similarity = scipy.stats.wasserstein_distance(laplacian1[:k], laplacian2[:k])

        # TODO: calculate Embedding similarity

        Embed_similarity = self._feature_similarity(Embed_G1, Embed_G2)
        similarity = 0.1 * Topology_similarity + (1-lamba) * Embed_similarity

        return similarity

    def _diversity_matrix(self, select_idx, sample_ratio=0.2):
        clusters = [idx for idx in range(self.args.cluster_number)]
        sample_number = math.ceil(self.args.cluster_number*sample_ratio)

        print('Diversity calculation')
        rec, rec_feature, rec_topo = {}, {}, {}
        G1 = self.sg_graph[select_idx]
        Embed_G1 = self.sg_features[select_idx]

        # diversity measurement of topology and embedding
        # TODO: embedding similarity calculation of hyperedge
        tic = time.time()
        for idx in tqdm(clusters):
            if idx != select_idx:
                Embed_G2 = self.sg_features[idx]
                rec_feature[idx] = self._feature_similarity(Embed_G1, Embed_G2) * 100
        toc = time.time()
        print('Time usage of feature diversity measurement: {}'.format(toc-tic))

        # TODO: topology similarity calculation of hyperedge
        tic = time.time()
        for idx in tqdm(clusters):
            if idx != select_idx:
                G2 = self.sg_graph[idx]
                laplacian1 = nx.spectrum.laplacian_spectrum(G1)
                laplacian2 = nx.spectrum.laplacian_spectrum(G2)

                k1 = self._select_k(laplacian1)
                k2 = self._select_k(laplacian2)

                k = min(k1, k2)

                Topology_similarity = scipy.stats.wasserstein_distance(laplacian1[:k], laplacian2[:k])

                rec_topo[idx] = Topology_similarity * 0.1

        toc = time.time()
        print('Time usage of topology diversity measurement: {}'.format(toc - tic))

        # TODO: diversity-driven sampling
        self.rec_topo = rec_topo
        self.rec_feature = rec_feature
        for idx in rec_topo.keys():
            rec[idx] = rec_topo[idx] + rec_feature[idx]

        rec_rank = sorted(rec.items(), key=lambda x: x[1], reverse=True)
        tmp = [select_idx]
        tmp.extend([f[0] for f in rec_rank[:(sample_number-1)]])

        return tmp

    def _diversity_matrix_update(self, select_idx, updated_embedding, epoch=None, sample_ratio=0.2):
        clusters = [idx for idx in range(self.args.cluster_number)]
        sample_number = math.ceil(self.args.cluster_number*sample_ratio)

        Embed_G1 = updated_embedding[select_idx]
        rec, rec_feature_update = {}, {}
        tic = time.time()

        for idx in tqdm(clusters):
            if idx != select_idx:
                Embed_G2 = updated_embedding[idx]
                rec_feature_update[idx] = self._feature_similarity(Embed_G1, Embed_G2) * 100
        toc = time.time()

        # TODO: diversity-driven sampling
        rec_topo = self.rec_topo
        rec_feature = self.rec_feature
        if epoch == None:
            for idx in rec_topo.keys():
                rec[idx] = rec_topo[idx] + (rec_feature[idx] + rec_feature_update[idx])/2
        else:
            None

        rec_rank = sorted(rec.items(), key=lambda x: x[1], reverse=True)
        tmp = [select_idx]
        tmp.extend([f[0] for f in rec_rank[:(sample_number-1)]])

        return tmp