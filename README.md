# Gaussian Infer

Proof of concept implementation for a Gaussian programming language with exact first-order conditioning, as described in ["Compositional Semantics for Probabilistic Programs with Exact Conditioning"](https://arxiv.org/abs/2101.11351). 

We provide two implementations, one using Python+`numpy`, the other using F#+`MathNet.Numerics`.

# Initial example: Gaussian regression

```python
g = Infer() # instantiate inference engine

# some regression data
xs = [1.0, 2.0, 2.25, 5.0, 10.0]
ys = [-3.5, -6.4, -4.0, -8.1, -11.0]

plt.plot(xs,ys,'ro')

# make Gaussian variables for slope and y-intercept
a = g.N(0,10)
b = g.N(0,10)
f = lambda x: a*x + b

# condition on the observations
for (x,y) in zip(xs,ys):
    g.condition(f(x), y + g.N(0,0.1))

# predicted means
ypred = [ g.mean(f(x)) for x in xs ]
plt.plot(xs,ypred)

plt.show()
```

![Gaussian regression](https://raw.githubusercontent.com/damast93/GaussianInfer/master/plot_regression.png)

# Structure

The implementation concerns an abstract type `rv` of Gaussian random variables, which can be combined linearly

```f#
type rv = 
    static member ( + ) : rv * rv -> rv       // sum of random variables
    static member ( * ) : float * rv -> rv    // scaling only with constants
    static member ( * ) : rv * float -> rv
    static member FromConstant : float -> rv  // numbers as constant random variables
    (...)
```

The inference engine maintains a prior over all random variables. New variables can be allocated using the `Normal` method and can be conditioned on equality.

```F#
type Infer =
    member Normal : float * float -> rv    // fresh normal variable with specified mean and variance
    member Condition : rv * rv -> unit     // condition on equality
    member Marginals : rv list -> Gaussian // marginals of a list of variable
    member Mean : rv -> float              // mean of an rv
    (...)
```

 The typed version of the Gaussian regression examples reads

```F#
open MathNet.Numerics.LinearAlgebra
open FSharp.Charting
open Infer

let make_infer() = 
    let infer = Infer()
    infer, infer.Normal, fun v w -> infer.Condition(v,w) 

let bayesian_regression_example() = 
    let infer, normal, (=.=) = make_infer()

    let xs = [1.0; 2.0; 2.25; 5.0; 10.0]
    let ys = [-3.5; -6.4; -4.0; -8.1; -11.0]

    let a = normal(0.0, 1.0)
    let b = normal(0.0, 1.0)
    let f x = a*x + b

    for (x, y) in List.zip xs ys do
      f(x) =.= y + normal(0.0, 0.1)

    let points = Chart.Point (List.zip xs ys)
    let regression_line = Chart.Line [ for x in xs -> (x, infer.Mean(f(x))) ]

    Chart.Combine [points; regression_line] |> Chart.Show 
```

# Implementation

The Python implementation features a class `Gauss` which represents an affine-linear map with Gaussian noise. This is a morphism in the category **Gauss** described by [[Fritz'20]](https://www.sciencedirect.com/science/article/abs/pii/S0001870820302656), and functions like `then` and `tensor` are provided to compose Gaussian maps like in a symmetric monoidal category. The class `Infer` represents the inference engine.

In the F# implementation, we do not model full Gaussian maps but only distributions. For conditioning of Gaussians, we directly implement the explicit formula from [Fritz'20] using the Moore-Penrose Pseudoinverse. Of course, in realistic code, you don't really want to compute inverses or Schur complements.

# Further Examples

## Kálmán filter

We implement a 1-dimensional [Kálmán filter](https://en.wikipedia.org/wiki/Kalman_filter) for predicting the movement of some object, say a plane, from noisy measurements. 

```python
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
```

![1D Kalman filter](https://raw.githubusercontent.com/damast93/GaussianInfer/master/plot_kalman.png)

## Gaussian Process Regression (Kriging)

We write [Gaussian process regression](https://en.wikipedia.org/wiki/Kriging) using the [rbf kernel](https://en.wikipedia.org/wiki/Radial_basis_function_kernel):

```python
# rbf kernel
rbf = lambda x,y: -0.2 * np.exp(-(x-y)**2)

def gp(xs, kernel):
    K = [[ kernel(x1,x2) for x1 in xs] for x2 in xs ]
    return G.N(np.array(K))

# build GP prior
xs = np.linspace(-5, 5, 100)
ys = g.from_dist(gp(xs, rbf))

# condition exactly on observed datapoints
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
```

![GP Regression](https://raw.githubusercontent.com/damast93/GaussianInfer/master/plot_gp.png)