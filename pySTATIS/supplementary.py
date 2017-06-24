from __future__ import print_function
import numpy as np
from .helpers import gen_affinity_input


def add_suptable_STATIS(Xsup, data, P, D, M, U, theta, n_datasets):

    QSup = Xsup.data_std_.T.dot(M.dot(P.dot(np.diag(1 / D))))
    FSup = Xsup.data_std_.dot(QSup)

    csup = []
    for i in range(n_datasets):
        csup.append(np.sum(Xsup.affinity_ * data[i].affinity_))

    GSup = np.atleast_2d(csup).dot(U[:, 0] * 1 / np.sqrt(theta[0]))
    ASup = 1 / np.sqrt(theta[0]) * GSup[0] * 1 / np.sum(U[:, 0])

    return QSup, FSup, GSup, ASup
