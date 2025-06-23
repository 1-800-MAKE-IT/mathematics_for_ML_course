import numpy as np
import numpy.linalg as la
np.set_printoptions(suppress=True)

#function to generate our micro model of the internet
def generate_internet(n) :
    c = np.full([n,n], np.arange(n))
    c = (abs(np.random.standard_cauchy([n,n])/2) > (np.abs(c - c.T) + 1)) + 0
    c = (c+1e-10) / np.sum((c+1e-10), axis=0)
    return c

def demo():

    #this matrix represents the transformation that you apply to a vector r (that represents a webpage) to simulate a timestep (where the user moves pages).
    # Each column is a website, A through to F. where there is a value, it means the webpage in that column has a link to the other respective website
    L = np.array([[0,   1/2, 1/3, 0, 0,   0 ],
                [1/3, 0,   0,   0, 1/2, 0 ],
                [1/3, 1/2, 0,   1, 0,   1/2 ],
                [1/3, 0,   1/3, 0, 1/2, 1/2 ],
                [0,   0,   0,   0, 0,   0 ],
                [0,   0,   1/3, 0, 0,   0 ]])

    #if this was a 3x3 matrix:
    """
    L = np.array([
            [L(a→a), L(b→a), L(c→a)],
            [L(a→b), L(b→b), L(c→b)],
            [L(a→c), L(b→c), L(c→c)],
        ])
    """
    #columns = the probability of leaving a website for any other website, and sum to one
    #rows = how likely you are to enter a given website from another, does not always sum to one

    #after an abritarily large number of timesteps, the only eigenvalue we care about is 1 due to the probabilistic nature
    #hence to get the pagerank we could just jump straight to calculating eigenvectors and values, but this is not scalable for large matrices
    
    def get_pagerank_with_eigenvalues(L):

        eVals, eVecs = la.eig(L) # Gets the eigenvalues and vectors
        order = np.absolute(eVals).argsort()[::-1] # Orders them by their eigenvalues
        eVals = eVals[order]
        eVecs = eVecs[:,order]

        r = eVecs[:, 0] # Sets r to be the principal eigenvector
        return 100 * np.real(r / np.sum(r)) # Make this eigenvector sum to one, then multiply by 100 Procrastinating Pats
    
    print(f'''Using eigenvalues and vectors we calculated that the pagerank looks like: {get_pagerank_with_eigenvalues(L)}. This should represent C, D, A, F, B, E''')

    def get_pagerank_iteratively():

        #first start by assuming equal distribution of users across our websites on the internet
        r = 100 * np.ones(6) / 6 # Sets up this vector (6 entries of 1/6 × 100 each)

        #keep running our r(i+1)=Lr(i) iterative equation until it converges:
        r = 100 * np.ones(6) / 6 # Sets up this vector (6 entries of 1/6 × 100 each)
        lastR = r
        r = L @ r

        i = 0
        while la.norm(lastR - r) > 0.01 :
            lastR = r
            r = L @ r
            i += 1

        print(str(i) + " iterations to convergence.")

        return r
    
    print(get_pagerank_iteratively())

    #however there is an issue - if we get a loop, such as a self connection, users can be simulated to get stuck in there,
    #so we need a probability that a user just types in a random URL 
    #we add this by including d. d = 1 means it has no effect, d = 0 means users will always just navigate to a random URL, hence all webpages have equal probability
    
    #we'll define a new internet matrix with a webpage that only has a self link going out wards
    L2 = np.array([[0,   1/2, 1/3, 0, 0, 0, 0 ],
               [1/3, 0,   0,   0, 1/2, 0, 0 ],
               [1/3, 1/2, 0,   1, 0,   1/3, 0 ],
               [1/3, 0,   1/3, 0, 1/2, 1/3, 0 ],
               [0,   0,   0,   0, 0,   0, 0 ],
               [0,   0,   1/3, 0, 0,   0, 0 ],
               [0,   0,   0,   0, 0,   1/3, 1 ]])
    
    def get_pagerank_interatively_with_dampening(d = 0.7):

        M = d * L2 + (1-d)/7 * np.ones([7, 7]) # np.ones() is the J matrix, with ones for each entry.

        r = 100 * np.ones(7) / 7 # Sets up this vector (6 entries of 1/6 × 100 each)

        lastR = r
        r = M @ r
        i = 0
        while la.norm(lastR - r) > 0.01 :
            lastR = r
            r = M @ r
            i += 1

        print(str(i) + " iterations to convergence.")

        return r
    
    print(get_pagerank_interatively_with_dampening())

demo()


# GRADED FUNCTION
# Complete this function to provide the PageRank for an arbitrarily sized internet.
# I.e. the principal eigenvector of the damped system, using the power iteration method.
# (Normalisation doesn't matter here)
# The functions inputs are the linkMatrix, and d the damping parameter - as defined in this worksheet.
# (The damping parameter, d, will be set by the function - no need to set this yourself.)

def pageRank(linkMatrix, d = 0.5):
    #main function that was graded
    n = linkMatrix.shape[0]

    M = d * linkMatrix + (1-d)/n * np.ones([n, n]) # np.ones() is the J matrix, with ones for each entry.

    r = 100 * np.ones(n) / n # Sets up this vector (6 entries of 1/6 × 100 each)

    lastR = r
    r = M @ r
    i = 0
    while la.norm(lastR - r) > 0.01 :
        lastR = r
        r = M @ r
        i += 1

    print(str(i) + " iterations to convergence.")

    return r

