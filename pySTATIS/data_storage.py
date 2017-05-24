from __future__ import print_function
import numpy as np
from scipy.stats.mstats import zscore


class STATISData(object):
    def __init__(self, X, ID, normalize=('zscore', 'norm_one'), col_names=None, row_names=None):
        """
        X: input variables for a single entity
        ID: ID of the entity; can be a set
        col_names, row_names: labels for rows and columns
        normalize: normalization method to use (None, 'zscore', 'double_center')
        """

        self.data = X
        self.ID = ID
        self.n_var = X.shape[1]

        if col_names is None:
            self.col_names = ['col_%s' % str(i).zfill(5) for i in range(X.shape[1])]
        else:
            self.col_names = col_names

        if row_names is None:
            self.row_names = ['row_%s' % str(i).zfill(5) for i in range(X.shape[0])]
        else:
            self.row_names = col_names

        self.normalize(method=normalize)
        self.cross_product()

    def normalize(self, method=None):

        if method is 'None':
            print("Not performing any normalizations")
            self.data_std = self.data
        else:
            temp = self.data
            for m in method:
                if m is 'zscore':
                    temp = zscore(temp, axis=0, ddof=1)

                elif m is 'norm_one':
                    temp = temp / np.linalg.norm(temp)

                else:
                    print("Method not implemented, use 'zscore' and/or 'double_center'")

                self.data_std = temp

    def cross_product(self):

        self.crossp = self.data_std.dot(self.data_std.T)