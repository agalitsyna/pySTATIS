'''
Python implementation of STATIS
'''

from __future__ import print_function

import numpy as np
from scipy.sparse.linalg import eigs
from scipy.stats.mstats import zscore

class STATISData(object):
    def __init__(self, X, ID, ev = None, similarity = 'cross_product', groups = ['group_1','group_2'], normalize=('zscore', 'norm_one'), col_names=None, row_names=None):
        """
        X: input variables for a single entity
        ID: ID of the entity; can be a set
        ev: eigenvalues of the X columns, in case that X are principal components
        col_names, row_names: labels for rows and columns
        normalize: normalization method to use (None, 'zscore', 'double_center')
        """

        self.data = X
        self.ev = ev
        self.ID = ID
        self.n_var = X.shape[1]
        self.groups = groups
        self.data_std_ = None

        if col_names is None:
            self.col_names = ['col_%s' % str(i).zfill(5) for i in range(X.shape[1])]
        else:
            self.col_names = col_names

        if row_names is None:
            self.row_names = ['row_%s' % str(i).zfill(5) for i in range(X.shape[0])]
        else:
            self.row_names = col_names

        self.normalize(method=normalize)
        """
        if similarity is 'cross_product':
            self.cross_product()
        elif similarity is 'covariance':
            self.covariance()
        elif similarity is 'double_center':
            self.double_center()
        """

        if ev is not None:
            self.data_scaled_ = self.data_std_ * self.ev
        else:
            self.data_scaled_ = self.data_std_

    def normalize(self, method=None):

        if method is 'None':
            print("Not performing any normalizations")
            self.data_std_ = self.data
        else:
            temp = self.data
            for m in method:
                if m is 'zscore':
                    temp = zscore(temp, axis=0, ddof=1)

                elif m is 'norm_one':
                    temp = temp / np.linalg.norm(temp)

                else:
                    print("Method not implemented, use 'zscore' and/or 'double_center'")

                self.data_std_ = temp

    def cross_product(self):

        self.crossp_ = self.data_std.dot(self.data_std.T)

    def covariance(self):

        self.cov_ = np.cov(self.data_std, rowvar = False)

    def double_center(self):

        assert self.data_std.shape[0] == self.data_std.shape[1], "Error: double centering requires a square matrix (correlation or covariance)"

        cmat = np.eye(self.data_std.shape[0]) - np.ones_like(self.data_std) * data_std.shape[0]
        self.dc_crossp_ = 0.5*cmat.dot(self.data_std.dot(cmat))


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

        self.groups_ = None
        self.ugroups_ = None
        self.n_groupings_ = None

        self.factor_scores_ = None
        self.partial_factor_scores_ = None
        self.col_indices_ = None

        self.contrib_obs_ = None
        self.contrib_var_ = None
        self.contrib_dat_ = None
        self.partial_inertia_dat_ = None
        self.contrib_grp_ = None

    def gen_affinity_input(self):

        for d in self.data:
            d.cross_product()


    def fit(self, data):

        """
        Main method to run STATIS on input example_data. Input example_data must be a list of STATISData objects.
        
        :param data: List of STATISdata objects
        """

        self.data = data
        self.n_datasets = len(self.data)
        self.n_observations = self.data[0].data.shape[0]

        # Pre-process
        self.gen_affinity_input()
        self.get_ids()
        self.get_groups()
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

    def get_groups(self):

        """
        Extracts the information about groups membership from the datasets.
        
        :return: 
        """

        self.groups_ = []
        for g in self.data:
            self.groups_.append(g.groups)
        self.groups_ = np.array(self.groups_)

        self.ugroups_ = []
        for i in range(self.groups_.shape[1]):
            self.ugroups_.append(np.unique(self.groups_[:, i]))

        self.n_groupings_ = len(self.ugroups_)

    def get_ids(self):

        self.ids_ = []

        for i in self.data:
            self.ids_.append(i.ID)

    def rv_pca(self):

        """
        Get weights for tables by calculating their similarity.
        
        :return: 
        """

        print("Running Rv-PCA...")
        C = np.zeros([self.n_datasets, self.n_datasets])

        for i in xrange(self.n_datasets):
            for j in xrange(0, self.n_datasets):
                C[i, j] = np.sum(self.data[i].crossp_ * self.data[j].crossp_)

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
        self.X_scaled_ = np.concatenate([self.data[i].data_scaled for i in range(self.n_datasets)], axis=1)

    def get_col_indices(self):

        """
        Returns dict(s) that maps IDs to columns 
        
        :return: 
        """

        self.col_indices_ = []
        c = 0
        for i, u in enumerate(self.ids_):
            self.col_indices_.append(np.arange(c, c + self.data[i].n_var))
            c += self.data[i].n_var

        self.grp_indices_ = []
        for i, ug in enumerate(self.ugroups_):
            ginds = []
            for g in ug:
                ginds.append(np.concatenate(map(self.col_indices_.__getitem__, np.where(self.groups_[:,i] == g)[0])))
            self.grp_indices_.append(ginds)

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

        self.partial_factor_scores_ = []

        for i, val in enumerate(self.col_indices_):
            self.partial_factor_scores_.append(np.inner(self.X_scaled_[:, val], self.Q_[val, :].T))

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

class dualSTATIS(STATIS):

    def gen_affinity_input(self):

        for d in self.data:
            d.covariance()