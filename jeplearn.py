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


def inertia(X, labels, centers):
    """Return the inertia of X corresponding to its labels and cluster centers
    
    Input:
        X: an array of shape (N,d)
        labels: an array of shape (N,)  
            that records the labels in (0, ..., k-1) of each sample 
        centers: an array of shape (k,d)  
            that records the cluster centers
    """
    k = centers.shape[0]
    return sum(np.sum((X[labels==i,:] - centers[i])**2)  for i in range(k))


def pca(X, r=2):
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


def mds(X, r=2, n_iter=100, verbose=0):
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


def kmeans(X, k, init="random"):
    """k-means clustering algorithm
    
    Input:  
        X: an array of shape (N,d)  
            rows for samples and columns for features
        k: number of clusters
        init: "random" or an array of shape (k,d)
            if "random", k points are chosen randomly from X as the initial cluster centers  
            if an array, the array is used as the initial cluster centers
        
    Output:
        (y_new, centers)
        y_new: an array of shape (N,)  
            that records the labels in (0, ..., k-1) of each sample 
        centers: an array of shape (k,d)  
            that records the cluster centers
            
    Example:
        mu = np.array([3,3])
        cov = np.eye(2)
        X = np.vstack([np.random.multivariate_normal(mu, cov, 100), 
                       np.random.multivariate_normal(-mu, cov, 100)])
        y_new,centers = kmeans(X, 2)
    """
    N,d = X.shape
    
    ### initialize y and center
    if isinstance(init, np.ndarray):
        centers = init.copy()
    elif init == "random":
        inds = np.random.choice(np.arange(N), k, replace=False)
        centers = X[inds, :]
    else:
        raise TypeError("init can only be a NumPy array or 'random'")

    dist = dist_mtx(X, centers)
    y_new = dist.argmin(axis=1)
    
    while True:        
        ### compute the new centers
        for i in range(k):
            mask = (y_new == i)
            centers[i] = X[mask].mean(axis=0)
        
        ### generate the new y_new
        dist = dist_mtx(X, centers)
        y_last = y_new.copy()
        y_new = dist.argmin(axis=1)
        
        if np.all(y_last == y_new):
            break

    return y_new, centers


def dbscan(X, eps, min_samples, draw=False):
    """DBSCAN clustering algorithm
    
    Input:  
        X: an array of shape (N,d)  
            rows for samples and columns for features
        eps: the radius used for finding neighborhood
        min_samples: a sample is considered as a core sample  
            if its epsilon-ball contains at least `min_sample` samples  
            (including itself)
        draw: boolean, return a illustrative figure or not

    Output:
        `(y_new, core_indices, fig)`  or `(y_new, core_indices)` depending on `draw` or not
        y_new: an array of shape (N,)  
            that records the labels in of each sample, where -1 stands for a noise 
        core_indices: an array of shape (n_core_samples,) that stores the indices of the core samples
        fig: an illustrative figure showing the data points, DFS tree, core samples, and noises

    Example:
        mu = np.array([3,3])
        cov = np.eye(2)
        X = np.vstack([np.random.multivariate_normal(mu, cov, 100), 
                       np.random.multivariate_normal(-mu, cov, 100)])
        y_new,core_indices,fig = dbscan(X, 1, 5, draw=True)
    """
    N,d = X.shape
    dist = dist_mtx(X)
    
    ### find core samples
    adj = (dist <= eps)
    core_mask = (adj.sum(axis=1) >= min_samples)
    core_indices = np.where(core_mask)[0]
    nbrhoods = [np.where(adj[i])[0] for i in range(N)]
    
    ### Run DFS to label each vertex
    y_new = -np.ones((N,), dtype=int)
    label_num = 0
    tree = []
    for i in range(N):
        if y_new[i] == -1 and core_mask[i]:
            stack = [i]
            while stack != []:
                j = stack.pop()
                if y_new[j] == -1:
                    tree.append((i,j))
                    y_new[j] = label_num
                    if core_mask[j]:
                        stack += list(nbrhoods[j])
            label_num += 1
    
    if draw:
        fig = plt.figure()
        for i,j in tree:
            plt.plot(*zip(X[i], X[j]), c='b', zorder=-1)
        plt.scatter(X[:,0], X[:,1], c=y_new, cmap='viridis')
        components = X[core_indices, :]
        plt.scatter(components[:,0], components[:,1], c='r', s=10)
        noise = X[y_new == -1]
        plt.scatter(noise[:,0], noise[:,1], c='k', s=100, marker='x')
        plt.close()
        return y_new, core_indices, fig
    
    return y_new, core_indices