import unittest
import numpy as np
from solution.mandelbrot_solution import mandelbrot, julia

class TestFractal(unittest.TestCase):
    def test_mandelbrot_shape(self):
        img = mandelbrot(width=100, height=80, max_iter=10)
        self.assertEqual(img.shape, (80, 100))
        self.assertTrue(np.all(img >= 0))

    def test_julia_shape(self):
        img = julia(0.355 + 0.355j, width=50, height=40, max_iter=5)
        self.assertEqual(img.shape, (40, 50))
        self.assertTrue(np.all(img >= 0))

if __name__ == '__main__':
    unittest.main()