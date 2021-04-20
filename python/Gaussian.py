import numpy as np
import scipy.linalg as linalg

# pick a consistent value from a list
def consistently(xs):
    vals = [ x for x in xs if x is not None ]
    assert(len(vals)>0)
    x = vals[0]
    assert(all(y == x for y in vals))
    return x


# implementation of a stochastic function with Gaussian noise
# (Reference [Fritz, Advances in Mathematics'2020])
class Gauss:
    
    def __init__(self, dom, cod, A=None,b=None,Sigma=None):      
        self.dom = dom
        self.cod = cod
        
        self.A = A if A is not None else np.zeros((cod,dom))
        self.b = b if b is not None else np.zeros((cod,1))
        self.Sigma = Sigma if Sigma is not None else np.zeros((cod,cod))


    # Condition on the first n variables 
    # cond : (a -> n + k) -> (a + n -> k) 
    def cond(self, n):
        m = self.cod-n
        SigmaX = self.Sigma[0:n,0:n]
        SigmaY = self.Sigma[n:,n:]
        SigmaYX = self.Sigma[n:,0:n]
        M = self.A[0:n,:]
        N = self.A[n:,:]
        s = self.b[0:n]
        t = self.b[n:]
        
        # scipy.linalg.schur
        SXi = np.linalg.pinv(SigmaX)
        D = SigmaYX.dot(SXi)
        A = np.block([D, N - D.dot(M)])
        b = t - D.dot(s)
        S = SigmaY - D.dot(SigmaYX.T)
        return Map(A,b,S)      
    
    def dot(self, other):
        A2 = self.A.dot(other.A)
        b2 = self.A.dot(other.b) + self.b
        Sigma2 = self.A.dot(other.Sigma).dot(self.A.T) + self.Sigma
        (m,n) = A2.shape
        return Gauss(n,m,A2,b2,Sigma2)
    
    def then(self,other):
        return other.dot(self)

    # cond_last : (a -> n + k) -> (a + k -> n)
    def cond_last(self, k):
        return self.then(swap(self.cod - k,k)).cond(k)
        
        
    def disintegrate(self, f):
        n = self.cod
        comp = self.then(copy(n)).then(tensor(f,eye(n)))
        return comp.cond(f.cod)
    
    def sample(self):
        assert(self.dom == 0)
        n = self.cod
        return np.random.multivariate_normal(self.b.reshape(n), self.Sigma)
    
    def __repr__(self):
        return f"<Gaussian map {self.dom} -> {self.cod}>\nA = \n{self.A}\nb = \n{self.b}\nSigma = \n{self.Sigma}"

def Map(A=None,b=None,Sigma=None):
    dom = consistently([ A.shape[1] if A is not None else None ])
    cod = consistently([ A.shape[0] if A is not None else None,
                         b.shape[0] if b is not None else None,
                         Sigma.shape[0] if Sigma is not None else None,
                         Sigma.shape[1] if Sigma is not None else None ])
    return Gauss(dom,cod,A,b,Sigma)

def N(arg):
    if type(arg) is int:
        n = arg
        return Gauss(0,n,None,None,np.eye(n))
    else:
        Sigma = arg
        (n,m) = Sigma.shape
        assert(n == m)
        return Gauss(0,n,None,None,Sigma)

def const(c):
    return Gauss(0,1,None,np.array([[c]]))

# tensor : (n -> m, n' -> m') -> n*n' -> m*m'
def tensor2(f,g):
    A = linalg.block_diag(f.A,g.A)
    b = np.block([[f.b],[g.b]])
    Sigma = linalg.block_diag(f.Sigma,g.Sigma)
    return Map(A,b,Sigma)

def tensor(*fs):
    total = eye(0)
    for f in fs:
        total = tensor2(total, f)
    return total

# copy : n -> n*n
def copy(n):
    A = np.block([[np.eye(n)],[np.eye(n)]])
    return Map(A)

# eye : n -> n
def eye(n):
    return Map(np.eye(n))

# dis : n -> 0
def dis(n):
    return Gauss(n, 0)

# swap : m+n -> n+m
def swap(m,n):
    A = np.block([
      [np.zeros((n,m)), np.eye(n)],
      [np.eye(m), np.zeros((m,n))]
    ])
    return Map(A)

# in_support : (0 -> n) -> n -> bool
# x is in the support of N(mu,sigma) iff the distance from x to im(Sigma) is < eps
# alternative implementation: just try solving Sigma z = x - b; cholesky on Sigma is efficient.
def in_support(dist, x, eps=1e-5):
    y = x - dist.b
    A = dist.Sigma
    Aplus = np.linalg.pinv(A)
    proj = A.dot(Aplus.dot(y)) # this is the projection on the column space of A
    d = np.linalg.norm(y - proj)
    if d >= eps: print("Distance to support: %f" % d)
    return d < eps