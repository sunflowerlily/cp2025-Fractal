import unittest
import numpy as np
from solution.fractal_dimension_solution import box_counting, generate_sierpinski_points

class TestFractalDimension(unittest.TestCase):
    def test_box_counting(self):
        data = generate_sierpinski_points(level=3)
        epsilons = np.array([8, 4, 2, 1])
        N = box_counting(data, epsilons)
        self.assertEqual(len(N), len(epsilons))
        self.assertTrue(np.all(N > 0))

if __name__ == '__main__':
    unittest.main()