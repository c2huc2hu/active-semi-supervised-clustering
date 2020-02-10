# Stands for either penalized or pairwise constrained kmeans
# Penalize constraint breaking with a weight penalty

import numpy as np
from scipy.spatial.distance import cdist

from active_semi_clustering.exceptions import EmptyClustersException
from .constraints import preprocess_constraints

class PCKMeans:
    def __init__(self, n_clusters=3, max_iter=100, n_init=10, w=1):
        '''

        w - penalty for breaking a constraint
        '''
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.w = w

    def fit(self, X, y=None, ml=[], cl=[]):
        best_score = np.inf
        self.cluster_centers_ = None
        self.labels_ = None

        ml_graph, cl_graph, neighborhoods = preprocess_constraints(ml, cl, X.shape[0])

        for _ in range(self.n_init):
            # Initialize centroids
            cluster_centers = self._initialize_cluster_centers(X, neighborhoods)

            # Repeat until convergence
            for iteration in range(self.max_iter):
                # Assign clusters
                try:
                    labels, score = self._assign_clusters(X, cluster_centers, ml, cl, self.w, self.w)
                except EmptyClustersException:
                    print('hit empty clusters')
                    break

                # Estimate means
                prev_cluster_centers = cluster_centers
                cluster_centers = self._get_cluster_centers(X, labels)

                if score < best_score:
                    self.cluster_centers_, self.labels_ = cluster_centers, labels
                    best_score = score

                # Check for convergence
                difference = (prev_cluster_centers - cluster_centers)
                converged = np.allclose(difference, np.zeros(cluster_centers.shape), atol=1e-6, rtol=0)

                if converged: break

        return self

    def _initialize_cluster_centers(self, X, neighborhoods):
        neighborhood_centers = np.squeeze(np.array([X[neighborhood].mean(axis=0) for neighborhood in neighborhoods]))
        neighborhood_sizes = np.array([len(neighborhood) for neighborhood in neighborhoods])

        if len(neighborhoods) > self.n_clusters:
            # Select K largest neighborhoods' centroids
            cluster_centers = neighborhood_centers[np.argsort(neighborhood_sizes)[-self.n_clusters:]]
        else:
            if len(neighborhoods) > 0:
                cluster_centers = neighborhood_centers
            else:
                cluster_centers = np.empty((0, X.shape[1]))

            # FIXME look for a point that is connected by cannot-links to every neighborhood set

            if len(neighborhoods) < self.n_clusters:
                remaining_cluster_centers = X[np.random.choice(X.shape[0], self.n_clusters - len(neighborhoods), replace=False), :]
                cluster_centers = np.concatenate([cluster_centers, remaining_cluster_centers])


        return cluster_centers

    def _assign_clusters(self, X, cluster_centers, ml, cl, w_m, w_c):
        # This version works slightly differently from the version in datamole's package
        # The original incrementally adds points, and can use labels generated so far when applying constraints.
        # This version labels everything at once with no constraints, then applies constraints based on that mapping

        # This method works fine for datasets with few constraints relative to the number of points,
        # and is at least an order of magnitude faster than the original. To get an even better approximation,
        # we iterate the tentative labelling and constraint-applying step.

        # This version doesn't apply transitivity to constraints.

        n = X.shape[0]
        k = len(cluster_centers)

        # Label without constraints
        initial_weights = cdist(X, cluster_centers, 'euclidean') # n x k 
        tentative_labels = initial_weights.argmin(axis=1)

        for i in range(2):
            # Apply must-link constraints
            ml_weights = np.zeros((n, k))
            for (i, j) in ml:
                ml_weights[i, tentative_labels[j]] += 1
                ml_weights[j, tentative_labels[i]] += 1

            # Apply cannot link constraints
            cl_weights = np.zeros((n, k))
            for (i, j) in cl:
                cl_weights[i, tentative_labels[j]] += 1
                cl_weights[j, tentative_labels[i]] += 1

            # Label
            weights = initial_weights - w_m * ml_weights + w_c * cl_weights
            tentative_labels = weights.argmin(axis=1)
            score = weights.min(axis=1).sum()
        labels = tentative_labels

        # Handle empty clusters
        # See https://github.com/scikit-learn/scikit-learn/blob/0.19.1/sklearn/cluster/_k_means.pyx#L309
        # TODO: Vanilla kmeans doesn't specify a procedure for this, but we can create a cluster out of 
        n_samples_in_cluster = np.bincount(labels, minlength=self.n_clusters)
        empty_clusters = np.where(n_samples_in_cluster == 0)[0]

        if len(empty_clusters) > 0:
            # print("Empty clusters")
            raise EmptyClustersException

        return labels, score

    def _get_cluster_centers(self, X, labels):
        return np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)]).squeeze()
