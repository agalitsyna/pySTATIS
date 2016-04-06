'''
Python implementation of STATIS
'''

import numpy as np
from scipy.sparse.linalg import eigs

def normalize_tables(X, type='default'):

    # Normalize each input table

    print "Normalizing tables..."

    from scipy.stats.mstats import zscore

    K = len(X)
    Xn = []

    if type == 'default':
        print "Z-scoring the columns of each matrix..."
        for k in range(K):
            X[k] = zscore(X[k],axis=0,ddof=1)
            Xn.append(X[k]/np.sqrt(np.sum(np.power(X[k],2))))

    if type == 'double_center':
        print "Double-centring the matrices..."
        for i in range(K):

            assert X[i].shape[0] != X[i].shape[1]

            N = X[i].shape[1]
            C = np.eye(N) - np.ones([N,N])/N
            Xn.append(np.dot(C*0.5, np.dot(X[i], C)))

    if type == 'none':
        print "Not performing any manipulations on input matrices"
        Xn = X

    return Xn

def lhstack(X):
    # Stack horizontally list of numpy arrays

    print "Stacking matrices horizontally..."

    K = len(X)
    Xs = X[0]

    for k in range(1,K):
        Xs = np.hstack([Xs, X[k]])

    return Xs

def get_mass_weight(X):
    # Get masses (M) for each observation (row), equal for each by default
    # Get weights (A) for each variable (column)

    print "Getting masses for rows and weights for columns..."

    K = len(X)
    Js = [x.shape[1] for x in X]

    # Get masses
    m = np.ones(X[0].shape[0])/X[0].shape[0]

    # Get similarity matrix
    print "Computing Z..."

    Z = np.vstack([np.dot(x, x.T).flatten() for x in X])

    print "Computing C..."

    C = np.dot(Z,Z.T)

    # Decompose similarity matrix to get table weights
    print "Decomposing similarity matrix..."

    [t, u] = eigs(C)
    t = np.real(t)
    u = np.real(u)
    u_r = u[:,0]/sum(u[:,0])
    a = np.concatenate([np.repeat(u_r[i],Js[i]) for i in range(K)])

    return m, a

def gsvd(X, m, a):

    print "Performing GSVD on the horizontally concatenated matrix..."

    M = np.diag(m)
    A = np.diag(a)

    Xt = np.dot(np.sqrt(M),np.dot(X, np.sqrt(A)))

    [P, D, Q] = np.linalg.svd(Xt, full_matrices=False)

    Mp = np.power(m,-0.5)
    Ap = np.power(a,-0.5)

    Pt = np.dot(np.diag(Mp),P)
    Qt = np.dot(np.diag(Ap),Q.T)
    D = np.diag(D)

    return Pt, D, Qt

def contrib(N, Nv, P, D, Q, m, a, n_comps = 3):
    # Calculate contributions of observations, variables and tables
    #
    # N - vector with number of rows, columns and tables in input data
    # Nv - vector with number if variables for each table
    # P - left singular vectors from GSVD
    # D - singular values from GSVD
    # Q - right singular vectors from GSVD
    # m - vector of masses of observations
    # a - vector of variable weights
    # n_comps - number of latent variables to compute contributions

    print "Calculating contributions of rows, columns and tables..."

    F = np.dot(P,D)[:,0:n_comps]

    # observations

    c_o = np.zeros([N[0],n_comps])

    for l in range(n_comps):
        for i in range(N[0]):
            c_o[i,l] = m[i] * F[i,l]**2 / D[l,l]**2

    # variables

    c_v = np.zeros([N[1], n_comps])

    for l in range(n_comps):
        for j in range(N[1]):
            c_v[j,l] = a[j] * Q[j,l]**2

    # tables

    c_t = np.zeros([N[2], n_comps])
    inds = np.concatenate([np.repeat(i, Nv[i]) for i in range(len(Nv))])

    for l in range(n_comps):
        for k in range(N[2]):
            c_t[k,l] = np.sum(c_v[inds==k,l])

    # partial intertia for tables

    I = c_t * np.diag(D)[range(n_comps)]**2

    return F, c_o, c_v, c_t, I

def project_back(X,Q, path = None, fname = 'bp_' ):
    # Projects individual tables into consensus

    import os
    import numpy as np

    if path == None:
        path = os.path.getcwd()

    N = len(X)
    Js = [x.shape[1] for x in X]
    inds = np.concatenate([np.repeat(i, Js[i]) for i in range(len(Js))])

    Fi = []

    for i in range(N):
        if os.path.isfile(os.path.join(path, fname + str(i).zfill(3))):
	  print "File already exists for subject %d" % i
          continue
        print "Reconstruction %d of %d" % (i, N)
        rec = np.dot(X[i],Q[inds == i,])
        np.save(os.path.join(path, fname + str(i).zfill(3)), rec)

    return path

def add_table(X, M, sres):
    # Projects data from new table onto consensus
    #
    # X - new table
    # sres - statis results

    Qs =  np.dot(X.T, np.dot(M, np.dot(sres['P'], np.linalg.inv(sres['D']))))

    Fs = np.dot(X, Qs)

    return Fs

def statis(X, fname='statis_results.npy'):
    # Main STATIS function

    Xn = normalize_tables(X, type = 'none')
    Xs = lhstack(Xn)
    m, a = get_mass_weight(X)
    [P, D, Q] = gsvd(Xs,m,a)
    Nv = [x.shape[1] for x in X]
    F, c_o, c_v, c_t, I = contrib([Xs.shape[0],Xs.shape[1],len(X)], Nv, P, D, Q, m, a, n_comps = 10)

    statis_res = dict(F=F, PI=I, C_rows = c_o, C_cols= c_v, C_tabs=c_t, P = P, D = D, Q = Q, M = m, A = a)

    np.save(fname, statis_res)

    return statis_res
