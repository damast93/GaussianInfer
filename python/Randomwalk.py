import numpy as np
import matplotlib.pyplot as plt

from Infer import Infer

#%% Condition on equality

# Model:
#   X, Y, Z ~ N(0,1) iid
#   X = Y
#   Y = Z

g = Infer()

x = g.N(50, 10*10)
y = g.N(x, 5*5)
g.condition(y, 40)


#%%%

y = [0]*100
y[0] = g.N(0,0.01)

obs = { 20: 15, 40: -10, 60: -15, 80: 5 }

for i in range(1,100):
    y[i] = y[i-1] + g.N(0,1)
for i in obs:
    g.condition(y[i],obs[i])
        
# for i in range(1,100):
#     if i in obs: 
#         y[i] = obs[i]
#         g.condition(y[i-1], g.N(y[i],1))
#     else:
#         y[i] = y[i-1] + g.N(0,1)

posterior = g.marginals(*y)
S2 = posterior.Sigma

for k in range(100):
    plt.plot(posterior.sample(), alpha=0.1)
    
plt.ylim([-30,30])