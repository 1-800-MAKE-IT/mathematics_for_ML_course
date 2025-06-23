from week_5_pagerank_algorithm.main import pageRank
import numpy as np
import numpy.linalg as la
np.set_printoptions(suppress=True)
import time

def get_pagerank_with_eigenvalues(L, d=0.5):
    """
    Calculate PageRank using the eigenvalue method with dampening factor.
    """
    n = L.shape[0]
    # Apply the same dampening as in pageRank function
    M = d * L + (1-d)/n * np.ones([n, n])
    
    eVals, eVecs = la.eig(M) # Gets the eigenvalues and vectors
    order = np.absolute(eVals).argsort()[::-1] # Orders them by their eigenvalues
    eVals = eVals[order]
    eVecs = eVecs[:,order]

    r = eVecs[:, 0] # Sets r to be the principal eigenvector
    return 100 * np.real(r / np.sum(r)) # Make this eigenvector sum to one, then multiply by 100

def generate_internet(n) :
    c = np.full([n,n], np.arange(n))
    c = (abs(np.random.standard_cauchy([n,n])/2) > (np.abs(c - c.T) + 1)) + 0
    c = (c+1e-10) / np.sum((c+1e-10), axis=0)
    return c

def test_20_dimensions():
    """
    Test both PageRank methods on a 20x20 link matrix:
    1. Compare results to ensure they match
    2. Compare performance (time to calculate)
    """
    L = generate_internet(20)
    
    # Time eigenvalue method
    start_eigen = time.time()
    eigen_result = get_pagerank_with_eigenvalues(L)
    eigen_time = time.time() - start_eigen
    print(f"\nEigenvalue method took: {eigen_time:.6f} seconds")
    
    # Time iterative method
    start_iter = time.time()
    iterative_result = pageRank(L)
    iter_time = time.time() - start_iter
    print(f"Iterative method took: {iter_time:.6f} seconds")
    print(f"Speed difference: {'eigenvalue' if eigen_time < iter_time else 'iterative'} method is {max(eigen_time, iter_time)/min(eigen_time, iter_time):.2f}× faster")
    
    # Compare results (with tolerance for floating point precision)
    print(f"\nEigenvalue result: {eigen_result}")
    print(f"Iterative result: {iterative_result}")
    
    # Use approximate comparison due to floating-point precision differences
    assert np.allclose(eigen_result, iterative_result, atol=1e-2)

def test_10_dimensions():
    """
    Test both PageRank methods on a 10×10 link matrix:
    1. Compare results to ensure they match
    2. Compare performance (time to calculate)
    """
    L = generate_internet(10)
    
    # Time eigenvalue method
    start_eigen = time.time()
    eigen_result = get_pagerank_with_eigenvalues(L)
    eigen_time = time.time() - start_eigen
    print(f"\nEigenvalue method took: {eigen_time:.6f} seconds")
    
    # Time iterative method
    start_iter = time.time()
    iterative_result = pageRank(L)
    iter_time = time.time() - start_iter
    print(f"Iterative method took: {iter_time:.6f} seconds")
    print(f"Speed difference: {'eigenvalue' if eigen_time < iter_time else 'iterative'} method is {max(eigen_time, iter_time)/min(eigen_time, iter_time):.2f}× faster")
    
    # Compare results (with tolerance for floating point precision)
    print(f"\nEigenvalue result: {eigen_result}")
    print(f"Iterative result: {iterative_result}")
    
    # Use approximate comparison due to floating-point precision differences
    assert np.allclose(eigen_result, iterative_result, atol=1e-2)

def test_5_dimensions():
    """
    Test both PageRank methods on a 5x5 link matrix:
    1. Compare results to ensure they match
    2. Compare performance (time to calculate)
    """
    L = generate_internet(5)
    
    # Time eigenvalue method
    start_eigen = time.time()
    eigen_result = get_pagerank_with_eigenvalues(L)
    eigen_time = time.time() - start_eigen
    print(f"\nEigenvalue method took: {eigen_time:.6f} seconds")
    
    # Time iterative method
    start_iter = time.time()
    iterative_result = pageRank(L)
    iter_time = time.time() - start_iter
    print(f"Iterative method took: {iter_time:.6f} seconds")
    print(f"Speed difference: {'eigenvalue' if eigen_time < iter_time else 'iterative'} method is {max(eigen_time, iter_time)/min(eigen_time, iter_time):.2f}× faster")
    
    # Compare results (with tolerance for floating point precision)
    print(f"\nEigenvalue result: {eigen_result}")
    print(f"Iterative result: {iterative_result}")
    
    # Use approximate comparison due to floating-point precision differences
    assert np.allclose(eigen_result, iterative_result, atol=1e-2)