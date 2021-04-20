namespace Infer

open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.Distributions

type Gaussian = { mu : Matrix<float> ; sigma : Matrix<float> }
  with
    member this.Dimension = this.sigma.RowCount

module Gaussian = 
   let empty = { mu = DenseMatrix.zero 0 1; sigma = DenseMatrix.zero 0 0 }
   let normal mean var = { mu = matrix [[ mean ]]; sigma = matrix [[ var ]] }

   let tensor (a : Gaussian) (b : Gaussian) = 
     let (m,n) = (a.Dimension, b.Dimension) in { 
      mu    = DenseMatrix.ofMatrixList2 [[ a.mu ]; [ b.mu ]]
      sigma = DenseMatrix.ofMatrixList2 [[ a.sigma; DenseMatrix.zero m n ]; [ DenseMatrix.zero n m; b.sigma ]]
    }

   let mapAffine (A : Matrix<float>, b) dist = {
     mu    = A * dist.mu + b
     sigma = A * dist.sigma * A.Transpose()
   }

   let condition n obs (dist : Gaussian) =
     let m = dist.Dimension - n
     let mu1 = dist.mu.[0..m-1,0..0]
     let mu2 = dist.mu.[m..m+n-1,0..0]
     let sigma11 = dist.sigma.[0..m-1,0..m-1]
     let sigma12 = dist.sigma.[0..m-1,m..m+n-1]
     let sigma22 = dist.sigma.[m..m+n-1,m..m+n-1]
     let sigma22inv = sigma22.PseudoInverse()
     {
       mu = mu1 + sigma12 * sigma22inv * (obs - mu2)
       sigma = sigma11 - sigma12 * sigma22inv * sigma12.Transpose()
     }


   // eigenvalues should be nonnegative, but are sometimes not because of rounding errors
   let safesqrt x = if x < 0.0 then 0.0 else sqrt x 
   
   let rnd = new Normal(0.0, 1.0)

   let sampler (dist : Gaussian) = 
     let n = dist.Dimension
     let evd = dist.sigma.Evd(Symmetricity.Symmetric)
     let sqrtD = DenseMatrix.ofDiagArray [| for i in 0..n-1 -> safesqrt (evd.D.[i,i]) |]
     let A = evd.EigenVectors * sqrtD
     fun () -> 
        let xs = matrix [ for i in 1..n -> [ rnd.Sample() ] ]
        dist.mu + A * xs

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
       let functionals = rvs |> List.map (toAffine state.Dimension)
       let A = DenseMatrix.ofMatrixList2 [ for (a,b) in functionals -> [ a ] ]
       let b = matrix [ for (a,b) in functionals -> [ b ] ]
       (A,b)   

   member this.FromDist(dist : Gaussian) = 
       let (m,n) = (state.Dimension, dist.Dimension)
       state <- Gaussian.tensor state dist
       [ for i in m..(m+n-1) -> Latent i ]

   member this.Normal(mean, var) =
       let m = state.Dimension
       state <- Gaussian.tensor state (Gaussian.normal mean var)
       Latent m

   member this.State = state

   member this.Marginals(rvs) = state |> Gaussian.mapAffine (toAffineMany rvs)

   member this.Condition(v,w) = 
     let vars = [ for i in 0..state.Dimension - 1 -> Latent(i) ]
     let joint = this.Marginals (vars @ [v-w])
     let posterior = joint |> Gaussian.condition 1 (matrix [[0.0]]) 
     state <- posterior

   member this.Mean(rv) = (this.Marginals [rv]).mu.[0,0]