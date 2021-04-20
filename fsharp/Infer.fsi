namespace Infer

open MathNet.Numerics.LinearAlgebra

type Gaussian = { mu : Matrix<float> ; sigma : Matrix<float> }

module Gaussian = 
   val empty : Gaussian
   val normal : float -> float -> Gaussian
   val tensor : Gaussian -> Gaussian -> Gaussian
   val sampler : Gaussian -> (unit -> Matrix<float>)

[<Sealed>]
type rv = 
    static member (+) : rv * rv -> rv
    static member (+) : float * rv -> rv
    static member (+) : rv * float -> rv
    static member (-) : rv * rv -> rv
    static member (-) : float * rv -> rv
    static member (-) : rv * float -> rv
    static member (~-) : rv -> rv
    static member ( * ) : float * rv -> rv
    static member ( * ) : rv * float -> rv
    static member FromConstant : float -> rv

[<Class>]
type Infer =
    new : unit -> Infer
    member FromDist : Gaussian -> rv list
    member Normal : float * float -> rv
    member Marginals : rv list -> Gaussian
    member Condition : rv * rv -> unit
    member Mean : rv -> float