import matplotlib.pyplot as plt
from Infer import Infer

#%% Condition on some equalities

# Model:
#   X, Y, Z ~ N(0,1) iid
#   X = Y
#   Y = Z

g = Infer()

x = g.N(0,1)
y = g.N(0,1)
z = g.N(0,1)

g.condition(x,y)
g.condition(y,z)

print(g.marginals(x,y))



#%% Bayesian regression

g = Infer()

xs = [1.0, 2.0, 2.25, 5.0, 10.0]
ys = [-3.5, -6.4, -4.0, -8.1, -11.0]

plt.plot(xs,ys,'ro')

a = g.N(0,10)
b = g.N(0,10)
f = lambda x: a*x + b

for (x,y) in zip(xs,ys):
    g.condition(f(x), y + g.N(0,0.1))

ypred = [ g.mean(f(x)) for x in xs ]
plt.plot(xs,ypred)

plt.show()

#%% Kalman filter

g = Infer()

xs = [ 1.0, 3.4, 2.7, 3.2, 5.8, 14.0, 18.0, 11.7, 19.5, 19.2]

x = [0] * len(xs)
v = [0] * len(xs)

# Initial parameters
x[0] = xs[0] + g.N(0,1)
v[0] = 1.0 + g.N(0,10)

for i in range(1,len(xs)):
    # Predict movement 
    x[i] = x[i-1] + v[i-1]
    v[i] = v[i-1] + g.N(0,0.75)
    
    # Make noisy observations
    g.condition(x[i] + g.N(0,1),xs[i])

plt.plot(xs,'ro')
plt.plot([ g.mean(x[i]) for i in range(len(xs)) ],'g')