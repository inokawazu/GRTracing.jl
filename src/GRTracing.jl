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

include("metrics.jl")
# include("tests.jl")
# include("ode.jl")
include("direction_setting.jl")
include("coordinate_change.jl")
include("rendering.jl")

# function render_pixels2headings(
#         met_renderer::MetricRenderer, pixel_coords;
#         time_final = 300
#     )

#     (;position, cam_matrix) = met_renderer

#     initial_headings = map(pixel_coords) do pixel_coord
#         (θ, ϕ) = uv_to_spherical(pixel_coord)

#         x = sin(ϕ) * cos(θ)
#         y = sin(ϕ) * sin(θ)
#         z = cos(ϕ)

#         cam_matrix * [x, y, z]
#     end
# end

# function test_render_pixels2headings(view_size::Integer)
#     mfunc = x -> schwarzschild_metric_iso_cart(x; rs = 1, c = 1)
#     mrend = MetricRenderer(mfunc, Float64[0, -100, 0], Float64[0,1,0])

#     # (mrend, (0.5, 0.5))
#     us = range(0.0, 1.0, length=view_size)
#     vs = range(0.0, 1.0, length=view_size)
#     view_coords = Iterators.product(us, vs)
#     render_pixels2headings(mrend, view_coords)
# end

end # module GRTracing
