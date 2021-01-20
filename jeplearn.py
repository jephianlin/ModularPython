import numpy as np
from scipy import linalg as LA

def dist_mtx(X, Y=None):
    """Return the distance matrix between rows of X and rows of Y
    
    Input:  
        X: an array of shape (N,d)
        Y: an array of shape (M,d)
            if None, Y = X
           
    Output:
        the matrix [d_ij] where d_ij is the distance between  
        the i-th row of X and the j-th row of Y
    """
    if isinstance(Y, np.ndarray):
        pass
    elif Y == None:
        Y = X.copy()
    else:
        raise TypeError("Y should be a NumPy array or None") 
    X_col = X[:, np.newaxis, :]
    Y_row = Y[np.newaxis, :, :]
    diff = X_col - Y_row
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    return dist

def PCA(X, r=2):
    """PCA dimensionality reduction  
    
    Input:  
        X: an array of shape (N,d)  
            rows for samples and columns for features
        r: target dimention (<= d)
        
    Output:
        an array of shape (N,r)
    
    Example:
        mu = np.array([3,4])
        cov = np.array([[1.1,1],
                        [1,1.1]])
        X = np.random.multivariate_normal(mu, cov, 100)
        X_new = PCA(X)
    """
    N,d = X.shape
    X = X - X.mean(axis= 0)
    C = 1. / N * X.T.dot(X)
    vals,vecs = LA.eigh(C, subset_by_index=[d-r,d-1])
    vals = vals[::-1] ### from large to small
    U = vecs[:,::-1] ### from large to small
    return X.dot(U)

def MDS(X, r=2, n_iter=100, verbose=0):
    """MDS dimensionality reduction
    
    Input:  
        X: an array of shape (N,d)  
            rows for samples and columns for features
        r: target dimention (<= d)
        n_iter: an integer for the number of iterations to run
        verbose: 0,1,2  
            shows more details when higher value
        
    Output:
        an array of shape (N,r)
        
    Example:
        mu = np.array([3,4])
        cov = np.array([[1.1,1],
                        [1,1.1]])
        X = np.random.multivariate_normal(mu, cov, 100)
        X_new = MDS(X, verbose=2)
    """
    N,d = X.shape
    goal = dist_mtx(X) ### dissimilarity in sklearn.manifold._mds.py
    
    X0 = np.random.randn(N, r)
    dis = dist_mtx(X0)
    stress = np.sum( (dis - goal)**2 ) / 2
    if verbose >= 2:
        print("Iter %3d: stress = %f"%(0, stress))
    
    Xk = X0
    for k in range(1, n_iter + 1):
        s = np.zeros_like(dis)
        mask = (dis != 0)
        s[mask] = 1 / dis[mask]
        B = -goal * s
        B[np.arange(N), np.arange(N)] = -B.sum(axis=1)
        Xk = 1. / N * B.dot(Xk) ### Vdagger B(Xk) Xk
        dis = dist_mtx(Xk)
        stress = np.sum( (dis - goal)**2 ) / 2
        if verbose >= 2:
            print("Iter %3d: stress = %f"%(k, stress))
            
    return Xk