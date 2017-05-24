'''
Python implementation of STATIS
'''

from __future__ import print_function

import numpy as np
from scipy.sparse.linalg import eigs


class STATIS(object):
    def __init__(self):

        self.data = None
        self.n_datasets = None
        self.n_observations = None

        self.A_ = None
        self.M_ = None
        self.X_ = None

        self.P_ = None  # Left singular vectors
        self.Q_ = None  # Right singular vectors
        self.D_ = None  # Singular values

        self.n_comps_ = None

        self.factor_scores_ = None
        self.partial_factor_scores_ = None
        self.col_indices_ = None

        self.contrib_obs_ = None
        self.contrib_var_ = None
        self.contrib_dat_ = None
        self.partial_inertia_dat_ = None

    def fit(self, data):

        """
        Main method to run STATIS on input data. Input data must be a list of STATISData objects.
        
        :param data: List of STATISdata objects
        """

        self.data = data
        self.n_datasets = len(self.data)
        self.n_observations = self.data[0].data.shape[0]

        # Pre-process
        self.rv_pca()
        self.get_A()
        self.get_M()
        self.stack_tables()

        # Decompose
        self.gsvd()

        # Get indices for each dataset
        self.get_col_indices()

        # Post-processing
        self.calc_factor_scores()
        self.calc_partial_factor_scores()
        self.calc_contrib_obs()
        self.calc_contrib_var()
        self.calc_contrib_dat()
        self.calc_partial_interia_dat()

        print("Done!")

    def rv_pca(self):

        """
        Get weights for tables by calculating their similarity.
        
        :return: 
        """

        print("Running Rv-PCA...")
        C = np.zeros([self.n_datasets, self.n_datasets])

        for i in xrange(self.n_datasets):
            for j in xrange(0, self.n_datasets):
                C[i, j] = np.sum(self.data[i].crossp * self.data[j].crossp)

        _, u = eigs(C, k=1)

        self.table_weights_ = np.real(u) / np.sum(np.real(u))

    def get_A(self):
        """
        Masses for tables.
        """
        print("Getting table masses...")
        a = np.concatenate([np.repeat(self.table_weights_[i], self.data[i].n_var) for i in range(self.n_datasets)])
        self.A_ = np.diag(a)

    def get_M(self):
        """
        Masses for observations. These are assumed to be equal.
        """
        print("Getting observation masses...")
        self.M_ = np.eye(self.n_observations) / self.n_observations

    def stack_tables(self):
        """
        Stacks preprocessed tables horizontally
        """

        print("Stacking tables...")

        self.X_ = np.concatenate([self.data[i].data_std for i in range(self.n_datasets)], axis=1)

    def get_col_indices(self):

        """
        Returns dict(s) that maps IDs to columns 
        
        :return: 
        """
        IDS = []
        for i in xrange(self.n_datasets):
            IDS.append(self.data[i].ID)
        self.IDS = IDS

        """
        self.col_indices_ = dict.fromkeys(IDS)
        c = 0
        for i, u in enumerate(IDS):
            self.col_indices_[u] = np.arange(c, c + self.data[i].n_var)
            c += self.data[i].n_var
        """

        self.col_indices_ = []
        c = 0
        for i, u in enumerate(IDS):
            self.col_indices_.append(np.arange(c, c + self.data[i].n_var))
            c += self.data[i].n_var

    def gsvd(self):

        print("Performing GSVD on the horizontally concatenated matrix...")

        self.Xw_ = np.dot(np.sqrt(self.M_), np.dot(self.X_, np.sqrt(self.A_)))

        [P, self.D_, Q] = np.linalg.svd(self.Xw_, full_matrices=False)

        Mp = np.power(np.diag(self.M_), -0.5)
        Ap = np.power(np.diag(self.A_), -0.5)

        self.P_ = np.dot(np.diag(Mp), P)
        self.Q_ = np.dot(np.diag(Ap), Q.T)
        self.ev_ = np.power(self.D_, 2)

        self.n_comps_ = len(self.D_)

    def calc_factor_scores(self):
        """
        Calculates factor scores for each observation.
        """

        print("Calculating factor scores for observations...")
        self.factor_scores_ = np.dot(self.P_, np.diag(self.D_))

    def calc_partial_factor_scores(self):

        """
        Projects individual scores onto the group-level component.
        """

        print("Calculating factor scores for datasets...")

        """
        self.partial_factor_scores_ = dict.fromkeys(self.IDS)

        for k, val in self.col_indices_.iteritems():
            self.partial_factor_scores_[k] = np.inner(self.X_[:, val], self.Q_[val, :].T)
        """

        self.partial_factor_scores_ = []

        for i, val in enumerate(self.col_indices_):
            self.partial_factor_scores_.append(np.inner(self.X_[:, val], self.Q_[val, :].T))

        self.partial_factor_scores_ = np.array(self.partial_factor_scores_)

    def calc_contrib_obs(self):

        """
        Contributions of observations (rows of the matrix)
        
        :return: 
        """
        print("Calculating contributions of observations...")

        self.contrib_obs_ = np.zeros([self.n_observations, len(self.D_)])

        for l in range(self.n_comps_):
            for i in range(self.n_observations):
                self.contrib_obs_[i, l] = self.M_[i, i] * self.factor_scores_[i, l] ** 2 / self.ev_[l]

    def calc_contrib_var(self):

        """
        Contributions of variables to a group-level components.
        :return: 
        """

        print("Calculating contributions of variables...")
        self.contrib_var_ = np.zeros([self.X_.shape[1], self.n_comps_])

        for l in range(self.n_comps_):
            for j in range(self.X_.shape[1]):
                self.contrib_var_[j, l] = self.A_[j, j] * self.Q_[j, l] ** 2

    def calc_contrib_dat(self):

        """
        Contributions of datasets/tables to the compromise.
        
        :return: 
        """

        print("Calculating contributions of datasets...")
        self.contrib_dat_ = np.zeros([self.n_datasets, self.n_comps_])

        for l in range(self.n_comps_):
            for i, k in enumerate(self.col_indices_):
                self.contrib_dat_[i, l] = np.sum(self.contrib_var_[k, l])

    def calc_partial_interia_dat(self):
        """
        Partial inertias for datasets/tables.
        
        :return: 
        """
        print("Calculating partial inertias for the datasets...")
        self.partial_inertia_dat_ = self.contrib_dat_ * self.ev_