import unittest
import numpy as np
from solution.fractal_custom_solution import generate_custom_fractal

class TestCustomFractal(unittest.TestCase):
    def test_generate_custom_fractal(self):
        points = generate_custom_fractal({"order":2, "scale":5})
        self.assertIsInstance(points, np.ndarray)
        self.assertEqual(points.shape[1], 2)
        self.assertTrue(points.shape[0] > 0)

if __name__ == '__main__':
    unittest.main()