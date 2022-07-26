import h5py
import numpy as np
from args import *

def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """
    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y

def main():
    with h5py.File(home_root+'/data/cluster.h5','r') as h5_file:
        X = h5_file['feats'][:]
    Y=pca(X,256)
    hh5 = h5py.File(home_root+'/data/cluster_pca_256.h5', 'w')
    hh5.create_dataset('feats', data = Y.data)
    hh5.close()
    print("dim reduce done")

if __name__ == '__main__':
    main()
