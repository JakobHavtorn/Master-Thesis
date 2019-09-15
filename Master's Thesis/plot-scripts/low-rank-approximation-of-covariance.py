import numpy as np
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import IPython


def low_rank_approximation(SVD=None, A=None, Aj=None, j=None, k=1):
    """Compute the rank k approximation of a matrix A given the matrix or its SVD
    """
    if not SVD:
        assert A is not None
        SVD = np.linalg.svd(A, full_matrices=False)
    U, S, V = SVD
    if Aj is not None:
        Ak = Aj
        ranks_to_add = range(j, k)
    else:
        Ak = np.zeros((U.shape[0], V.shape[1]))
        ranks_to_add = range(k)
    for i in ranks_to_add:
        Ak += S[i] * np.outer(U.T[i], V[i])
    return Ak


def save_matrix(A, filepath):
    """Save a matrix as a .pdf file.
    """
    f, a = plt.subplots()
    a.imshow(A, cmap='Greys_r')
    plt.axis('off')
    f.savefig(filepath + '.pdf', bbox_inches='tight')
    plt.close(f)


def process_matrix(A, ranks=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 18, 22, 30, 40, 50, 100]):
    """Process a matrix, computing its SVD for some ranks and plotting
    """
    # SVD
    SVD = np.linalg.svd(A, full_matrices=False)
    U, S, V = SVD
    # Save plot of singular values
    f, ax = plt.subplots()
    ax.plot(S)
    ax.set_xlabel('Singular value index')
    ax.set_ylabel('Singular value')
    f.savefig('../graphics/svd/' + matrix_name + '-svds.pdf', bbox_inches='tight')
    plt.close(f)
    f, ax = plt.subplots()
    ax.semilogy(S)
    ax.set_xlabel('Singular value index')
    ax.set_ylabel('Singular value')
    f.savefig('../graphics/svd/' + matrix_name + '-svds-log.pdf', bbox_inches='tight')
    plt.close(f)
    # Compute low rank approximation approximation and plot
    MSEs, Fnorm, rs = [], [], []
    for k in sorted(ranks):
        if k < A.shape[0]:
            if len(rs) == 0:
                Ak = low_rank_approximation(SVD, k=k)
            else:
                Ak = low_rank_approximation(SVD, Aj=Ak, j=j, k=k)
            MSEs.append(((A - Ak) ** 2).mean(axis=None))
            Fnorm.append(np.linalg.norm(A - Ak, 2))
            rs.append(k)
            save_matrix(Ak, '../graphics/svd/' + matrix_name + '-' + str(k))
            j = k
    # MSE of rank approx
    f, ax = plt.subplots()
    ax.plot(rs, MSEs)
    ax.set_xlabel('Rank of approximation, ' + r'$L$')
    ax.set_ylabel('MSE' + r'$(\mathbf{A}_L)$')
    f.savefig('../graphics/svd/' + matrix_name + '-MSE.pdf', bbox_inches='tight')
    f, ax = plt.subplots()
    ax.semilogy(rs, MSEs)
    ax.set_xlabel('Rank of approximation, ' + r'$L$')
    ax.set_ylabel('MSE' + r'$(\mathbf{A}_L)$')
    f.savefig('../graphics/svd/' + matrix_name + '-MSE-log.pdf', bbox_inches='tight')
    # Frobenius norm distance
    f, ax = plt.subplots()
    ax.plot(rs, Fnorm)
    ax.set_xlabel('Rank of approximation, ' + r'$L$')
    ax.set_ylabel(r'$|| \mathbf{A} - \mathbf{A}_L ||_2$')
    f.savefig('../graphics/svd/' + matrix_name + '-fnorm-distance.pdf', bbox_inches='tight')


if __name__ == '__main__':
    matplotlib.rcParams.update({'font.size': 12})
    # Load image
    matrix_name = '42'
    A = plt.imread(matrix_name + '.jpg', format='jpeg')
    A = A[0:499,:]
    save_matrix(A, '../graphics/svd/' + matrix_name + '-full')
    process_matrix(A)

    # Load image
    matrix_name = '324'
    A = plt.imread(matrix_name + '.jpg', format='jpeg')
    A = np.dot(A[..., :3], [0.299, 0.587, 0.114])
    save_matrix(A, '../graphics/svd/' + matrix_name + '-full')
    process_matrix(A)

    # Random covariance (small)
    np.random.seed(42)
    matrix_name = 'random-covariance-20x20'
    size = 20
    A = np.random.rand(size, size)
    A = np.dot(A, A.transpose())
    save_matrix(A, '../graphics/svd/' + matrix_name + '-full')
    process_matrix(A)

    # Random covariance (large)
    np.random.seed(42)
    matrix_name = 'random-covariance-600x600'
    size = 600
    A = np.random.rand(size, size)
    A = np.dot(A, A.transpose())
    save_matrix(A, '../graphics/svd/' + matrix_name + '-full')
    process_matrix(A)
