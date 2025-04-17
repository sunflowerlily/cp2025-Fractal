import unittest
import numpy as np
from solution.percolation_solution import generate_lattice, find_clusters, has_percolating_cluster

class TestPercolation(unittest.TestCase):
    def test_lattice_shape(self):
        lattice = generate_lattice(size=20, p=0.5, seed=1)
        self.assertEqual(lattice.shape, (20, 20))
        self.assertTrue(np.all((lattice == 0) | (lattice == 1)))

    def test_find_clusters(self):
        lattice = generate_lattice(size=10, p=0.7, seed=2)
        labeled, n = find_clusters(lattice)
        self.assertEqual(labeled.shape, (10, 10))
        self.assertTrue(n >= 1)

    def test_percolating(self):
        lattice = np.ones((5,5), dtype=int)
        labeled, n = find_clusters(lattice)
        perc = has_percolating_cluster(labeled)
        self.assertTrue(len(perc) > 0)

if __name__ == '__main__':
    unittest.main()