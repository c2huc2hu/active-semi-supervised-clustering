import unittest
import time

from sklearn import datasets, metrics
from sklearn.cluster import KMeans

from active_semi_clustering.semi_supervised.pairwise_constraints import PCKMeans
from active_semi_clustering.semi_supervised.pairwise_constraints.old_pckmeans import PCKMeans as PCKMeans_Old
from active_semi_clustering.active.pairwise_constraints import ExampleOracle, ExploreConsolidate, MinMax, Random

class IrisBase(unittest.TestCase):
    def setUp(self):
        self.X, self.y = datasets.load_iris(return_X_y=True)

class KMeansTest(IrisBase):
    '''Vanilla KMeans'''

    def test(self):
        print('x', self.X.shape)

        t0 = time.time()

        clusterer = KMeans(n_clusters=3, random_state=141342).fit(self.X)
        print('took', time.time() - t0, 'seconds')

        score = metrics.adjusted_rand_score(self.y, clusterer.labels_)
        print('Ran kmeans on iris, got score', score)

class PCKMeansTest(IrisBase):
    def test(self):
        oracle = ExampleOracle(self.y, max_queries_cnt=100)

        active_learner = Random(n_clusters=3)
        active_learner.fit(self.X, oracle=oracle)
        pairwise_constraints = active_learner.pairwise_constraints_

        t0 = time.time()
        clusterer = PCKMeans(n_clusters=3)
        clusterer.fit(self.X, ml=pairwise_constraints[0], cl=pairwise_constraints[1])
        print('took', time.time() - t0, 'seconds')

        score = metrics.adjusted_rand_score(self.y, clusterer.labels_)
        print('Ran pckmeans on iris, got score', score)
        self.assertTrue(score > 0.7)

        return score

class MinMaxTest(IrisBase):
    def test(self):
        oracle = ExampleOracle(self.y, max_queries_cnt=100)

        active_learner = MinMax(n_clusters=3)
        active_learner.fit(self.X, oracle=oracle)
        pairwise_constraints = active_learner.pairwise_constraints_

        t0 = time.time()
        clusterer = PCKMeans(n_clusters=3)
        clusterer.fit(self.X, ml=pairwise_constraints[0], cl=pairwise_constraints[1])
        print('n constraints:', len(pairwise_constraints[0]) + len(pairwise_constraints[1]))

        print('took', time.time() - t0, 'seconds')

        score = metrics.adjusted_rand_score(self.y, clusterer.labels_)
        print('Ran pckmeans + minmax on iris, got score', score)
        self.assertTrue(score > 0.7)

        return score

class OldPCKMeansTest(IrisBase):
    def test(self):
        return 
        oracle = ExampleOracle(self.y, max_queries_cnt=100)

        active_learner = Random(n_clusters=3)
        active_learner.fit(self.X, oracle=oracle)
        pairwise_constraints = active_learner.pairwise_constraints_

        t0 = time.time()
        clusterer = PCKMeans_Old(n_clusters=3)
        clusterer.fit(self.X, ml=pairwise_constraints[0], cl=pairwise_constraints[1])
        print('took', time.time() - t0, 'seconds')

        score = metrics.adjusted_rand_score(self.y, clusterer.labels_)
        print('Ran old pckmeans on iris, got score', score)
        self.assertTrue(score > 0.7)

        return score

if __name__ == "__main__":
    unittest.main()
