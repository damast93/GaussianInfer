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

#%% 1D-Kalman filter

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
plt.show()

#%% 2D-Kalman filter (airplane tracking)

g = Infer()

# some coordinates and times of observation
posx  = [ 1.0, 2.0, 3.0, 4.0,   6.0, 5.5, 5.0, 4.0 ]
posy  = [ 2.0, 3.0, 3.0, 3.8,   1.0, 0.0, 1.0, 1.2 ]
times = [ 0, 1, 2, 3,           7, 8, 9, 10 ]

N = 25
TMax = 12
dt = TMax / N

def index(time):
    return int(time / dt)

x, y = [0] * N, [0] * N
vx, vy = [0] * N, [0] * N

x[0] = posx[0] + g.N(0, 0.01)
y[0] = posy[0] + g.N(0, 0.01)

for i in range(1, N):
    ax = g.N(0, 0.1)
    ay = g.N(0, 0.1)
    
    vx[i] = vx[i-1] + ax*dt
    vy[i] = vy[i-1] + ay*dt
    
    x[i] = x[i-1] + vx[i]*dt
    y[i] = y[i-1] + vy[i]*dt

# condition on being close to observations
for i in range(0,len(times)):
    t = times[i]
    g.condition(x[index(t)], posx[i] + g.N(0, 0.02))
    g.condition(y[index(t)], posy[i] + g.N(0, 0.02))
   
plt.plot(posx,posy,'r*--', alpha=0.5)
 
trajectory = g.marginals(*(x + y))
for i in range(1,25):
    vals = trajectory.sample()
    xs,ys = vals[0:N], vals[N:]
    
    plt.plot(xs, ys, '--', color='green', alpha=0.1)

xmean = [ g.mean(xi) for xi in x ]
ymean = [ g.mean(yi) for yi in y ]

plt.plot(xmean, ymean, 'b-')
plt.show()

#%% A distribution over closed loops

g = Infer()

N = 25
dt = 1

x, y = [0] * N, [0] * N
vx, vy = [0] * N, [0] * N

x[0] = 0.0 + g.N(0, 0.01)
y[0] = 0.0 + g.N(0, 0.01)

# simulate some trajectories
for i in range(1, N):
    ax = g.N(0, 0.01)
    ay = g.N(0, 0.01)
    
    vx[i] = vx[i-1] + ax*dt
    vy[i] = vy[i-1] + ay*dt
    
    x[i] = x[i-1] + vx[i]*dt
    y[i] = y[i-1] + vy[i]*dt

# condition that trajectories return to their origin
g.condition(x[N-1], x[0])
g.condition(y[N-1], y[0])

trajectory = g.marginals(*(x + y))
for i in range(1,25):
    vals = trajectory.sample()
    xs,ys = vals[0:N], vals[N:]
    
    plt.plot(xs, ys, '-', alpha=0.25)

plt.show()
    