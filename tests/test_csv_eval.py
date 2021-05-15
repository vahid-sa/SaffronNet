import unittest
import numpy as np
from retinanet.csv_eval import compute_distance, prepare


class TestCSVEval(unittest.TestCase):
    """ Test Anchor's functions functionality
    """

    def test_prepare(self):
        data = {
            'a': np.array([2.1]),  # a: (N)
            'b': np.array([1.1, 1.2, 1.3, 1.4])  # b: (k)
        }

        a, b = prepare(data['a'], data['b'])  # prepare: (N, k)
        assert a.shape == b.shape
        assert a.shape[0] == data['a'].shape[0]
        assert a.shape[1] == data['b'].shape[0]

    def test_compute_distance(self):
        """ test compute distance
        """
        data = {
            'a': np.array([[2.1, 2.2, 23]]),  # a: (N)
            # b: (k)
            'b': np.array([[3.1, 3.2, 33], [4.1, 4.2, 22], [5.1, 5.2, 53]])
        }
        dxys, dangles = compute_distance(data['a'], data['b'])

        assert dxys.shape == dangles.shape

        assigned_annotation = np.argmin(dxys, axis=1)
        min_dxy = dxys[0, assigned_annotation]
        assert assigned_annotation == [0]

        assigned_annotation = np.argmin(dangles, axis=1)
        min_dangel = dangles[0, assigned_annotation]
        assert assigned_annotation == [1]
