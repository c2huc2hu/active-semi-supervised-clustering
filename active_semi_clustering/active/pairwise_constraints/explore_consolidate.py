import numpy as np
from scipy.spatial.distance import cdist

from .helpers import get_constraints_from_neighborhoods
from .example_oracle import MaximumQueriesExceeded


class ExploreConsolidate:
    def __init__(self, n_clusters=3, **kwargs):
        self.n_clusters = n_clusters

    def fit(self, X, oracle=None):
        '''X is a np array: num_examples x feature_size'''
        if oracle.max_queries_cnt <= 0:
            return [], []

        neighborhoods = self._explore(X, oracle)
        neighborhoods = self._consolidate(neighborhoods, X, oracle)

        self.pairwise_constraints_ = get_constraints_from_neighborhoods(neighborhoods)

        return self

    def _explore(self, X, oracle):
        '''
        Create k clusters by taking the furthest element and asking the oracle whether it is part of an existing cluster.        
        '''

        # Pick a random point in X
        n = X.shape[0]
        x = np.random.choice(n)

        # Add that point to traversed as our starting point
        traversed_idx = set([x])
        untraversed_idx = set(range(n)) - traversed_idx
        neighborhoods = [[x]]

        try:
            while len(neighborhoods) < self.n_clusters:

                # Find the untraversed point the furthest from its closest neighbour in traversed
                traversed = X[np.array(list(traversed_idx))]

                if len(untraversed_idx) <= self.n_clusters:
                    untraversed = X[np.array(untraversed_idx)]
                    mindists = cdist(untraversed, traversed, 'euclidean').min(axis=1)
                    farthest = np.argmax(mindists)
                else:
                    # Computing all the distances is expensive, so randomly sample self.n_clusters points.
                    index_map = np.random.choice(list(untraversed_idx), size=(self.n_clusters,), replace=False)
                    untraversed = X[index_map]
                    mindists = cdist(untraversed, traversed, 'euclidean').min(axis=1)
                    farthest = index_map[np.argmax(mindists)]

                # Find out if the farthest point is in a new neighborhood. This is equivalent to topic discovery
                for neighborhood in neighborhoods: # TODO: If we have many neighborhoods this costs a lot of queries. Instead, should rank neighbours by their distance to farthest
                    if oracle.query(farthest, neighborhood[0]):
                        neighborhood.append(farthest)
                        break
                else:
                    neighborhoods.append([farthest])
                traversed_idx.add(farthest)
                untraversed_idx.remove(farthest)

        # Break when oracle exceeds its limit of MaximumQueriesExceeded
        except MaximumQueriesExceeded:
            pass

        return neighborhoods

    def _consolidate(self, neighborhoods, X, oracle):
        '''
        Categorize each point until running out of oracle queries
        '''

        n = X.shape[0]

        traversed_idx = {elem for elem in neigh for neigh in neighborhoods}
        untraversed_idx = set(range(n)) - traversed_idx        

        while True:
            try:
                # Randomly select a point
                i = np.random.choice(list(untraversed_idx))

                # Find the closest neighbourhood
                cdist(X[i], X[np.array(untraversed_idx)]) 

                dists = cdist(X[i], representatives, 'euclidean') # might need to expand_dim X[i]
                sorted_neighborhoods = np.argsort(dists)

                for neigh_idx in sorted_neighborhoods:
                    if oracle.query(i, neighborhoods[neigh_idx]):
                        neighborhood.append(i)
                        break
                traversed_idx.add(i)
                untraversed_idx.remove(i)

            except MaximumQueriesExceeded:
                break

        return neighborhoods


# def dist(i, S, points):
#     # Find the smallest distance from points[i] to points[j] for any j. 

#     # Construct pairwise distance
#     distance_mtx = ((points - points.T) ** 2).sum(axis=-1)
#     return np.sqrt(distance_mtxi.min())

#     distances = np.array([np.sqrt(((points[i] - points[j]) ** 2).sum()) for j in S])
#     return distances.min()
