import numpy as np
import matplotlib.pyplot as plt

from Infer import Infer

#%% Condition on equality

# Model:
#   X, Y, Z ~ N(0,1) iid
#   X = Y
#   Y = Z

g = Infer()

x0 = g.N(0,1)
x1 = x0 + g.N(0,1)
x2 = x1 + g.N(0,1)
x3 = x2 + g.N(0,1)
x4 = x3 + g.N(0,1)

g.condition(x0,0)
g.condition(x4,0)

# g.condition(x,y)

print(g.marginals(x2))