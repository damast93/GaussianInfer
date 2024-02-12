import numpy as np

import Gaussian as Gauss

# data Term = Var name | Normal name | Const r | Mul r t | Add s t

counter = 0
def gensym():
    global counter
    counter += 1
    return counter

def invert(xs):
    return { x : i for (i,x) in enumerate(xs) }

class Term:
    
    def __init__(self, typ, **kwargs):
        self.__dict__ = kwargs
        self.typ = typ
        
    def var(name):
        return Term('var', name=name)
    
    def normal(name):
        return Term ('normal',name=name)
    
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
            return "Var@%i" % self.name
        elif self.typ == 'normal':
            return "Normal@%i" % self.name
        elif self.typ == 'add':
            return str(self.s) + ' + ' + str(self.t)
        elif self.typ == 'mul':
            return str(self.r) + '*(' + str(self.t) + ')'
        else:
            raise Exception("Bad case")
            
    def _repr_pretty_(self, p, cycle):
        p.text(str(self) if not cycle else '...')
   
def collect_normals(term):
    if   term.typ == 'normal': return { term.name }
    elif term.typ == 'const': return set()
    elif term.typ == 'var': return set()
    elif term.typ == 'add': return collect_normals(term.s).union(collect_normals(term.t))
    elif term.typ == 'mul': return collect_normals(term.t)
         

def normal():
    return Term.normal(gensym())

    
def eval_term_old(var_idx, term):
    n = len(var_idx)
    
    if term.typ == 'const':
        return Gauss.Gauss(n, 1, None, term.r, None)
    elif term.typ == 'var':
        proj = np.zeros((1,n))
        proj[0,var_idx[term.name]] = 1.0
        return Gauss.Gauss(n, 1, proj, None, None)
    elif term.typ == 'normal':
        return Gauss.Gauss(n, 1, None, None, np.array([[1.0]]))
   
# n variables, m normals -> Gauss(m,1), Gauss(n,1)     

def proj(n,i):
    p = np.zeros((1,n))
    p[0,i] = 1.0
    return p

add = Gauss.Gauss(2, 1, np.array([[1.0, 1.0]]))
        
def eval_term(var_idx, normal_idx, term):
    n = len(var_idx)
    m = len(normal_idx)
    
    if term.typ == 'const':
        return Gauss.Gauss(n, 1, None, term.r, None), Gauss.Gauss(m, 1)
    
    elif term.typ == 'var':
        return Gauss.Gauss(n, 1, proj(n, var_idx[term.name])), Gauss.Gauss(m, 1)
    
    elif term.typ == 'normal':
        return Gauss.Gauss(n, 1), Gauss.Gauss(m, 1, proj(m, normal_idx[term.name]))
    
    elif term.typ == 'mul':
        f, g = eval_term(var_idx, normal_idx, term.t)
        scale = Gauss.Gauss(1, 1, np.array([[term.r]]))
        return f.then(scale), g.then(scale)
   
    elif term.typ == 'add':
        f1, g1 = eval_term(var_idx, normal_idx, term.s)
        f2, g2 = eval_term(var_idx, normal_idx, term.t)
        return Gauss.copy(n).then(Gauss.tensor(f1, f2)).then(add), Gauss.copy(m).then(Gauss.tensor(g1, g2)).then(add)

def listwrap(xs):
    return [Term.lift(xs)] if isinstance(xs, (Term, float, int)) else [ Term.lift(x) for x in xs ]

from inspect import signature

def gauss(fn):
    sig = signature(fn)
    params = sig.parameters
    n = len(params)
    
    input_vars = [ gensym() for i in range(n) ]
    var_idx = invert(input_vars)
    
    ts = listwrap(fn(*[ Term.var(v) for v in input_vars]))
    k = len(ts)
    
    normals = list(set( normal for t in ts for normal in collect_normals(t) ))
    normal_idx = invert(normals)
    m = len(normals)
    
    morphism = Gauss.Gauss(n+m, 0)
    
    for t in ts:
        f, g = eval_term(var_idx, normal_idx, t)
        morphism = Gauss.copy(n+m).then(Gauss.tensor(morphism,Gauss.tensor(f,g).then(add)))
   
    return Gauss.tensor(Gauss.eye(n), Gauss.N(m)).then(morphism)

# translate a Gauss map back into a term
# TODO turn it into a function instead
def ungauss(F):
    n = F.dom
    k = F.cod
    A, b, Sigma = F.A, F.b, F.Sigma
    
    L = np.linalg.cholesky(Sigma)
    xs = [ Term.var(gensym()) for j in range(n) ]
    ys = [ Term.normal(gensym()) for i in range(k) ]
    
    zs = [None] * k
    for i in range(k):
        zs[i] = b[i,0] + sum(A[i,j]*xs[j] for j in range(n)) + sum(L[i,r]*ys[r] for r in range(k))
    return zs

# Example: Noisy measurement
f = lambda x, y: x + 2*y + 3*(normal() + normal())

def g(x,y):
    z = normal()
    return x + 2*y + 3*(z+z) + 5

@gauss
def ex():
    x = 50 + 10*normal()
    y = x + 5*normal()
    return y, x