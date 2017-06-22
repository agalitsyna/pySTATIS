import numpy as np

from .. import get_wine_data
from ..decomposition import rv_pca


def test_RvPCA():
    data = get_wine_data()
    n_datasets = len(data)

    expected_output = np.array([[0.10360263],
                                [0.10363524],
                                [0.10370834],
                                [0.09208477],
                                [0.08063234],
                                [0.09907428],
                                [0.09353886],
                                [0.08881811],
                                [0.1110871],
                                [0.12381833]])

    assert np.allclose(rv_pca(data, n_datasets), expected_output)