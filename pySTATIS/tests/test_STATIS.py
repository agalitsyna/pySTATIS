import numpy as np

from unittest import TestCase
from pySTATIS.wine_data import get_wine_data
from pySTATIS.core.decomposition import rv_pca

class TestSTATIS(TestCase):

    def test_RvPCA(self):
        data = get_wine_data()
        n_datasets = len(data)

        expected_output = np.array([[ 0.10360263],
                                   [ 0.10363524],
                                   [ 0.09208477],
                                   [ 0.10370834],
                                   [ 0.08063234],
                                   [ 0.09907428],
                                   [ 0.09353886],
                                   [ 0.08881811],
                                   [ 0.1110871 ],
                                   [ 0.12381833]])

        np.testing.assert_allclose(rv_pca(data, n_datasets), expected_output)


if __name__ == "__main__":
    unittest.main()
