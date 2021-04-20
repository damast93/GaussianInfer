import numpy as np

import Gaussian as Gauss

class InferenceError(Exception):
    def __init__(self, message):
        super(Exception, self).__init__(message)


# affine-linear term of Gaussians
# data Term = Var int | Const float | Mul float Term | Add Term Term 
class Term:
    
    def __init__(self, typ, **kwargs):
        self.__dict__ = kwargs
        self.typ = typ
        
    def var(index):
        return Term('var', index=index)
    
    def const(r):
        return Term('const', r=float(r))
    
    def lift(v):
        if isinstance(v, (int,float)):
            return Term.const(v)
        elif isinstance(v,Term):
            return v
        else:
            raise Exception("Unknown type")
        
    def __add__(self, other):
        return Term('add', s=self, t=Term.lift(other))
    
    def __radd__(self, other):
        return Term('add', s=self, t=Term.lift(other))
    
    def __mul__(self, other):
        return Term('mul', t=self, r=float(other))
    
    def __rmul__(self, other):
        return Term('mul', t=self, r=float(other))
    
    def __str__(self):
        if self.typ == 'const':
            return str(self.r)
        elif self.typ == 'var':
            return "<@%i>" % self.index
        elif self.typ == 'add':
            return str(self.s) + ' + ' + str(self.t)
        elif self.typ == 'mul':
            return str(self.r) + '*' + str(self.t)
        else:
            raise Exception("Bad case")
   
# Stateful inference engine
class Infer:
    
    def __init__(self):
        self.n = 0 # number of variables
        self.equations = []
        self.state = Gauss.N(0)
        
    def nu(self):
        # make new variable
        v = Term.var(self.n)
        
        # extend state 
        self.state = Gauss.tensor(self.state, Gauss.N(1))
        
        self.n += 1
        return v
    
    def N(self, mu,sigmasq):
        return np.sqrt(sigmasq)*self.nu() + mu 
    
    
    # add a multivariate distribution to the state
    def from_dist(self, dist):
        assert(dist.dom == 0)
        k = dist.cod
        
        self.state = Gauss.tensor(self.state, dist)
        
        vs = [ Term.var(i) for i in range(self.n, self.n+k) ]
        
        self.n += k
        return vs 
        
    # return a pair (A,b) where A : (1,n), b : float
    def eval_term_rec(self, term):
        n = self.n
        
        if term.typ == 'const':
            return (np.zeros((1,n)), term.r)
        elif term.typ == 'var':
            A = np.zeros((1,n))
            A[0,term.index] = 1.0
            return (A, 0.0)
        elif term.typ == 'add':
            (A1,b1) = self.eval_term_rec(term.s)
            (A2,b2) = self.eval_term_rec(term.t)
            return (A1 + A2, b1 + b2)
        elif term.typ == 'mul':
            (A,b) = self.eval_term_rec(term.t)
            r = term.r
            return (r * A, r * b)
        else:
            return Exception("Bad case")
        
    def eval_term(self, term):
        (A,b) = self.eval_term_rec(term)
        return Gauss.Map(A,np.array([[ b ]]))
    
    def marginals(self, *args):
        self.do_inference()
        
        n = self.n
        proj = Gauss.Map(np.zeros((0,n))) # n -> 0
        for t in args:
            proj = Gauss.copy(n).then(Gauss.tensor(proj, self.eval_term(t)))
        return self.state.then(proj)
    
    def mean(self, t):
        self.do_inference()
        d = self.state.then(self.eval_term(t))
        return d.b[0,0]
    
    def variance(self, t):
        self.do_inference()
        d = self.state.then(self.eval_term(t))
        return d.Sigma[0,0]
    
    def condition(self, s, t):
        self.equations.append((s,t))
        
    def do_inference(self):
        n_eqs = len(self.equations)
        
        if n_eqs > 0:
            n = self.n
            joint = Gauss.Map(np.zeros((0,n))) # n -> 0
            for (s,t) in self.equations:
                diff = self.eval_term(s + (-1.0)*t)
                joint = Gauss.copy(n).then(Gauss.tensor(joint, diff))
            
            zz = np.zeros((n_eqs,1))
            zeros = Gauss.Map(np.zeros((n_eqs,0)), zz) # 0 -> n_eqs
            
            if not Gauss.in_support(self.state.then(joint), zz):
                raise InferenceError("conditions cannot be satisfied")
            else:
                dis = self.state.disintegrate(joint)
                self.state = dis.dot(zeros)
                self.equations = []
            
            