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
    
    let regression_line_sample() =
        let sample = Gaussian.sampler (infer.Marginals [ a; b ]) ()
        let a0, b0 = sample.[0,0], sample.[1,0]
        Chart.Line [ for x in xs -> x, a0*x + b0 ]

    Chart.Combine ([points; regression_line] @ [ for i in 1..10 -> regression_line_sample() ]) |> Chart.Show 
  

let gp_example() = 
    let infer, normal, (=.=) = make_infer()

    let xs = [ -5.0 .. 0.1 .. 5.0 ]
    let observations = [(20, 1.0); (40, 2.0); (60, 3.0); (80, 0.0)]

    // rbf kernel
    let rbf x y = 0.2 * exp(-(x-y)*(x-y))

    let ys = infer.FromDist {
        mu    = matrix [ for x in xs -> [ 0.0 ] ]
        sigma = matrix [ for x in xs -> [ for y in xs -> rbf x y ]] 
      }

    for (i,y) in observations do
      ys.[i] =.= rv.FromConstant y

    let points = Chart.Point [ for i, y in observations -> (xs.[i], y) ]
    let regressionLine = Chart.Line [ for x, y in List.zip xs ys -> (x, infer.Mean y) ] 

    let sampler = infer.Marginals(ys) |> Gaussian.sampler
    let sample_graph() = 
        let graph = sampler()
        Chart.Line [ for x, y in List.zip xs (graph.AsColumnMajorArray() |> List.ofSeq) -> (x, y) ] 

    Chart.Combine  ([points; regressionLine]  @ [ for i in 1..10 -> sample_graph() ]) |> Chart.Show

bayesian_regression_example()
gp_example()