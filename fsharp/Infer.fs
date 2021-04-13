namespace Infer

open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.Distributions

type Gaussian = { dim : int; mu : Matrix<float> ; sigma : Matrix<float> }

module Gaussian = 
   let empty = { dim = 0; mu = DenseMatrix.zero 0 1; sigma = DenseMatrix.zero 0 0 }
   let normal mean var = { dim = 1; mu = matrix [[ mean ]]; sigma = matrix [[ var ]] }
   let tensor a b = 
     let (m,n) = (a.dim, b.dim) in { 
      dim   = m + n
      mu    = DenseMatrix.ofMatrixList2 [[ a.mu ]; [ b.mu ]]
      sigma = DenseMatrix.ofMatrixList2 [[ a.sigma; DenseMatrix.zero m n ]; [ DenseMatrix.zero n m; b.sigma ]]
    }
   let mapAffine (A : Matrix<float>, b) dist = {
     dim   = A.RowCount
     mu    = A * dist.mu + b
     sigma = A * dist.sigma * A.Transpose()
   }
   let condition n obs dist =
     let m = dist.dim - n
     let mu1 = dist.mu.[0..m-1,0..0]
     let mu2 = dist.mu.[m..m+n-1,0..0]
     let sigma11 = dist.sigma.[0..m-1,0..m-1]
     let sigma12 = dist.sigma.[0..m-1,m..m+n-1]
     let sigma22 = dist.sigma.[m..m+n-1,m..m+n-1]
     let sigma22inv = sigma22.PseudoInverse()
     {
       dim = m
       mu = mu1 + sigma12 * sigma22inv * (obs - mu2)
       sigma = sigma11 - sigma12 * sigma22inv * sigma12.Transpose() // Better formula for Schur complement
     }
   let sample dist = 
     let dist = new MatrixNormal(dist.mu, dist.sigma, DenseMatrix.identity 1)
     dist.Sample()

type rv = Const of float | Sum of rv * rv | Scale of float * rv | Latent of int
  with
    static member (+) (a : rv, b : rv) = Sum (a,b)
    static member (+) (a : float, b : rv) = Sum (Const a,b)
    static member (+) (a : rv, b : float) = Sum (a, Const b)
    static member ( * ) (r : float, a : rv) = Scale (r,a)
    static member ( * ) (a : rv, r : float) = Scale (r,a)
    static member (~-) (a : rv) = Scale(-1.0,a)
    static member (-) (a : rv, b : rv) = Sum(a,Scale(-1.0,b))
    static member (-) (a : float, b : rv) = Sum(Const a, Scale(-1.0,b))
    static member (-) (a : rv, b : float) = Sum(a, Const (-b))
    static member FromConstant c = Const c

type Infer() = 
   let mutable state = Gaussian.empty

   let rec toAffine n = function
   | Const c -> (DenseMatrix.zero 1 n, c)
   | Sum(v, w) -> 
       let (a1, c1) = toAffine n v
       let (a2, c2) = toAffine n w
       (a1 + a2, c1 + c2)
   | Scale(r, v) ->
       let (a, c) = toAffine n v 
       (r * a, r * c)
   | Latent(i) -> 
       (DenseMatrix.init 1 n (fun _ j -> if j = i then 1.0 else 0.0), 0.0)

   let toAffineMany rvs =
       let functionals = rvs |> List.map (toAffine state.dim)
       let A = DenseMatrix.ofMatrixList2 [ for (a,b) in functionals -> [ a ] ]
       let b = matrix [ for (a,b) in functionals -> [ b ] ]
       (A,b)   

   member this.FromDist(dist) = 
       let (m,n) = (state.dim, dist.dim)
       state <- Gaussian.tensor state dist
       [ for i in m..(m+n-1) -> Latent i ]

   member this.Normal(mean, var) =
       let m = state.dim
       state <- Gaussian.tensor state (Gaussian.normal mean var)
       Latent m

   member this.State = state

   member this.Marginals(rvs) = state |> Gaussian.mapAffine (toAffineMany rvs)

   member this.Condition(v,w) = 
     let vars = [ for i in 0..state.dim - 1 -> Latent(i) ]
     let joint = this.Marginals (vars @ [v-w])
     let posterior = joint |> Gaussian.condition 1 (matrix [[0.0]]) 
     state <- posterior

   member this.Mean(rv) = (this.Marginals [rv]).mu.[0,0]