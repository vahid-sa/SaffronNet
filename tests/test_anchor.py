import unittest
import numpy as np
from retinanet.anchor_utils import generate_anchors

class TestAnchor(unittest.TestCase):
    """ Test Anchor's functions functionality
    """

    def test_generate_anchors(self):
        """ test that it can generate corect base anchors
        """
        data = {
            'angle_split': 4,
            'num_variables': 3
        }

        num_variables = 3
        num_anchors = data['angle_split']
        result_anchors = np.zeros((num_anchors, num_variables))
        result_anchors[:, -1] = [0.0, 90.0, 180.0, 270.0]

        anchors = generate_anchors(**data)

        self.assertEqual(result_anchors.shape, anchors.shape)
        self.assertEqual(np.sum(result_anchors[:, :-1]), np.sum(anchors[:, :-1]))
        self.assertTrue(np.all(result_anchors[:, -1] == anchors[:, -1]))


if __name__ == '__main__':
    unittest.main()


