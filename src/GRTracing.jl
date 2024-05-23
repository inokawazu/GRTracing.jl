module GRTracing

using Colors
using FileIO
using LinearAlgebra
using OrdinaryDiffEq
# using Symbolics
using ThreadsX
using ForwardDiff
using FiniteDiff
using StaticArrays
using DiffEqPhysics

include("metrics.jl")
include("lagrangian.jl")
include("metric_accel.jl")
# include("ode.jl")
include("direction_setting.jl")
include("coordinate_change.jl")
include("rendering.jl")
include("tests.jl")

end # module GRTracing
