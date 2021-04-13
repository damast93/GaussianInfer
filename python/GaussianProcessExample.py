import numpy as np
import matplotlib.pyplot as plt

from Infer import Infer
import Gaussian as G

g = Infer()

# rbf kernel
rbf = lambda x,y: -0.2 * np.exp(-(x-y)**2)

def gp(xs, kernel):
    K = [[ kernel(x1,x2) for x1 in xs] for x2 in xs ]
    return G.N(np.array(K))

# build GP prior
xs = np.linspace(-5, 5, 100)
ys = g.from_dist(gp(xs, rbf))

# add observations
observations = [(20,1.0), (40, 2.0), (60, 3.0), (80, 0.0)]
for (i,yobs) in observations:
    g.condition(ys[i],yobs)

# plot sample functions
posterior = g.marginals(*ys)
for i in range(750):
    samples = posterior.sample()
    plt.plot(xs,samples,alpha=0.05)
    
# Plot means
means = np.array([ g.mean(y) for y in ys ])
plt.plot(xs, means, 'red')

# Standard deviation ± 3sigma
stds = np.array([ np.sqrt(np.abs(g.variance(y))) for y in ys ])
plt.plot(xs, means + 3*stds, 'blue', xs, means - 3*stds, 'blue')
