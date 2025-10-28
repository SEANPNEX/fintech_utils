import numpy as np

def ew_cov(mat, lam=0.97):
    """
        Exponentially weighted covariance matrix
        mat: m x n matrix, m observations, n variables
        lam: decay factor
        return: n x n ew covariance matrix
    """
    m, n = mat.shape
    # get the weight
    w = np.array([(1 - lam) * (lam ** (m - 1 - i)) for i in range(m)])
    w = w / sum(w)
    # get diag(w)
    W = np.diag(w)
    # calculate mu with mu = Rw
    mu = np.dot(mat.T, w)
    # calculate ew cov with EW = RWR^T - mumu^T
    ew_cov = mat.T @ W @ mat - np.outer(mu, mu)
    return ew_cov


# ew correlation matrix, lambda = 0.94
def ew_corr(mat, lam=0.94):
    """
        Exponentially weighted correlation matrix
        mat: m x n matrix, m observations, n variables
        lam: decay factor
        return: n x n ew correlation matrix
    """
    cov = ew_cov(mat, lam)
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    return corr

# near psd matrix
def near_psd_cov(A, epsilon = 0):
    """
        Near PSD covariance matrix
        A: m x m covariance matrix
        epsilon: small positive value to ensure positive semi-definiteness
        return: m x m near PSD covariance matrix
    """
    # get correlation matrix
    d = np.sqrt(np.diag(A))
    corr = A / np.outer(d, d)
    eig_vals, eig_vecs = np.linalg.eig(corr)
    eig_vals[eig_vals < epsilon] = epsilon
    corr_psd = (eig_vecs @ np.diag(eig_vals) @ eig_vecs.T)
    corr_psd = corr_psd / np.outer(np.sqrt(np.diag(corr_psd)), np.sqrt(np.diag(corr_psd)))
    near_A = np.outer(d, d) * corr_psd
    return near_A

def near_psd_corr(A, epsilon = 0):
    eig_vals, eig_vecs = np.linalg.eig(A)
    eig_vals[eig_vals < epsilon] = epsilon
    corr_psd = (eig_vecs @ np.diag(eig_vals) @ eig_vecs.T)
    corr_psd = corr_psd / np.outer(np.sqrt(np.diag(corr_psd)), np.sqrt(np.diag(corr_psd)))
    return corr_psd

# Higham coverance matrix
def higham_corr(A, max_iter=100, tol=1e-6):
    """
        Higham correlation matrix
        A: m x m covariance matrix
        max_iter: maximum number of iterations
        tol: tolerance for convergence
        return: m x m higham correlation matrix
    """
    n = A.shape[0]
    # symmetric
    A = (A + A.T) / 2
    X = A.copy()
    Y = np.zeros_like(A)
    for i in range(max_iter):
        R = X - Y
        eig_vals, eig_vecs = np.linalg.eig(R)
        eig_vals[eig_vals < 0] = 0
        X_new = eig_vecs @ np.diag(eig_vals) @ eig_vecs.T
        Y = X_new - R
        X_new[np.diag_indices(n)] = 1
        if np.linalg.norm(X_new - X, ord='fro') < tol:
            break
        X = X_new
    return X


def higham_cov(A, max_iter=100, tol=1e-6):
    """
        Higham covariance matrix
        A: m x m covariance matrix
        max_iter: maximum number of iterations
        tol: tolerance for convergence
        return: m x m higham covariance matrix
    """
    # get correlation matrix
    d = np.sqrt(np.diag(A))
    corr = A / np.outer(d, d)
    corr_psd = higham_corr(corr, max_iter, tol)
    near_A = np.outer(d, d) * corr_psd
    return near_A

def pca_cov(mat, var_explained=0.99):
    """
        PCA covariance matrix
        mat: m x m covariance matrix
        var_explained: variance explained threshold
        return: m x m PCA covariance matrix
    """
    eigvals, eigvecs = np.linalg.eigh(mat)
    total_var = np.sum(eigvals)
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]
    cum_var = np.cumsum(eigvals)
    num_components = np.searchsorted(cum_var, var_explained * total_var) + 1
    L = eigvecs[:, :num_components] @ np.diag(np.sqrt(eigvals[:num_components]))
    return L @ L.T