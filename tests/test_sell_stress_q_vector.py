import unittest

import numpy as np

import sell_stress as ss


class BuildQVectorTests(unittest.TestCase):
    def test_linear_vector_is_bounded_and_includes_endpoints(self):
        values = ss.build_q_vector("linear", 10.0)

        self.assertEqual(len(values), 200)
        self.assertLessEqual(len(values), ss.MAX_Q_POINTS_PER_SECURITY)
        self.assertGreater(values[0], 0)
        self.assertAlmostEqual(values[-1], 10.0)
        self.assertTrue(np.all(np.diff(values) > 0))

    def test_log_vector_preserves_small_positive_values(self):
        values = ss.build_q_vector("log", 10.0)

        self.assertEqual(len(values), 200)
        self.assertTrue(np.all(values > 0))
        self.assertTrue(np.all(np.diff(values) > 0))
        self.assertAlmostEqual(values[-1], 10.0)

    def test_custom_point_count_cannot_exceed_limit(self):
        with self.assertRaises(ValueError):
            ss.build_q_vector("linear", 10.0, q_points=ss.MAX_Q_POINTS_PER_SECURITY + 1)


if __name__ == "__main__":
    unittest.main()
