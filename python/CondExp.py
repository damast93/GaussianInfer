import numpy as np
import Gaussian as Gauss

# Conditioning-free reimplementation focusing on
# random variables and conditional expectations

# Q'n'D name generation
__counter = 0
def gensym():
    global __counter
    sym = f"@{__counter}"
    __counter += 1
    return sym

# affine-linear term of Gaussians
# data Rv = Var @index | Const float | Mul float Rv | Add Term Rv 
class Rv:
    
    def __init__(self, typ, **kwargs):
        self.__dict__ = kwargs
        self.typ = typ
        
    def var(ix):
        return Rv('var', index=ix)
    
    def const(r):
        return Rv('const', r=float(r))
    
    def standard_normal():
        return Rv.var(gensym())
    
    def normal(mean, std):
        return std * Rv.standard_normal() + mean
    
    def lift(v):
        if isinstance(v, (int,float)):
            return Rv.const(v)
        elif isinstance(v,Rv):
            return v
        else:
            raise Exception("Unknown type")
        
    def __add__(self, other):
        return Rv('add', s=self, t=Rv.lift(other))
    
    def __radd__(self, other):
        return Rv('add', s=self, t=Rv.lift(other))
    
    def __mul__(self, other):
        return Rv('mul', t=self, r=float(other))
    
    def __rmul__(self, other):
        return Rv('mul', t=self, r=float(other))
    
    def __str__(self):
        if self.typ == 'const':
            return str(self.r)
        elif self.typ == 'var':
            return self.index
        elif self.typ == 'add':
            return str(self.s) + ' + ' + str(self.t)
        elif self.typ == 'mul':
            return str(self.r) + '*' + str(self.t)
        else:
            raise Exception("Bad case")
    
    def __repr__(self):
        return str(self)
    
    #------------
    
    def mean(self):
        if self.typ == 'const':
            return self.r
        elif self.typ == 'var':
            return 0.0
        elif self.typ == 'add':
            return self.s.mean() + self.t.mean()
        elif self.typ == 'mul':
            return self.r * self.t.mean()
        else:
            raise Exception("Bad case")
         
    def cov_ix(self,ix):
        if self.typ == 'const':
            return 0.0
        elif self.typ == 'var':
            return (1.0 if self.index == ix else 0.0)
        elif self.typ == 'add':
            return self.s.cov_ix(ix) + self.t.cov_ix(ix)
        elif self.typ == 'mul':
            return self.r * self.t.cov_ix(ix)
        else:
            raise Exception("Bad case")  
                
    def cov(self,other):
        if self.typ == 'const':
            return 0.0
        elif self.typ == 'var':
            return other.cov_ix(self.index)
        elif self.typ == 'add':
            return self.s.cov(other) + self.t.cov(other)
        elif self.typ == 'mul':
            return self.r * self.t.cov(other)
        else:
            raise Exception("Bad case")

# covenient interface 
def mean(*xs):
    return np.array([[x.mean()] for x in xs])
            
def cov_v(xs,ys):
    m,n = len(xs), len(ys)
    Sigma = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            Sigma[i,j] = xs[i].cov(ys[j])
    return Sigma

def var(*xs):
    return cov_v(xs,xs)

def condexp_v(xs,ys):
    Sigma_XY = cov_v(xs,ys)
    Sigma_YY = cov_v(ys,ys)
    S = Sigma_XY @ np.linalg.pinv(Sigma_YY)
    EE = mean(*xs) + S @ (np.array([ys]).T + (-1.0) * mean(*ys))
    return EE

def E(x):
    return lambda *ys: condexp_v([x],ys)[0,0]

# Example ...

x = Rv.normal(0,1)
y = Rv.normal(x,1)

print(mean(y), var(y))

z = E(x)(y)
c = x + (-1)*z

# show that the residue c is orthgonal to y

print(var(c,y))