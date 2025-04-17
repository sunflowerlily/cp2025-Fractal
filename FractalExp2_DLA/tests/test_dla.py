import unittest
import numpy as np
from solution.dla_solution import dla_simulation

class TestDLA(unittest.TestCase):
    def test_dla_shape(self):
        grid = dla_simulation(size=21, n_particles=10, seed=42)
        self.assertEqual(grid.shape, (21, 21))
        self.assertTrue(np.any(grid))

    def test_dla_growth(self):
        grid = dla_simulation(size=31, n_particles=20, seed=123)
        self.assertGreater(np.sum(grid), 1)

if __name__ == '__main__':
    unittest.main()