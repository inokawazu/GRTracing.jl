module GRTracing

using Symbolics, LinearAlgebra, OrdinaryDiffEq, ThreadsX

"""
    schwarzschild_metric_function((t,x,y,z); rs, c = 1)
"""
function schwarzschild_metric_iso_cart((t,x,y,z); rs, c = 1)
    R = (sqrt∘sum)(x->x^2,(x,y,z))
    rm = (1 - rs/R)^2
    rp = (1 + rs/R)^2
    return [
     -rm/rp*c^2 0 0 0
     0 rp^2 0 0
     0 0 rp^2 0
     0 0 0 rp^2
    ]
end

"""
    geodesic_acceleration( metric_fun::Function )

    Gives the acceleration according to GR
"""
function geodesic_acceleration_func( metric::Function, dim::Integer )
    position = Symbolics.variables(:x, 0:dim-1)
    velocity = Symbolics.variables(:v, 0:dim-1)
    g = metric(position)
    ig = inv(g)
    dg = [
          Symbolics.derivative(g[i], c) 
          for i in eachindex(IndexCartesian(), g), c in position
         ]

    accel = [
     simplify(sum(
                  -1/2*ig[i, l] * (dg[n,l,m] + dg[m,l,n] - dg[m,n,l]) * velocity[n] * velocity[m]
                  for m in 1:dim, n in 1:dim, l in 1:dim
                 ))
     for i in 1:length(velocity)
    ]

    eval(build_function(accel, velocity, position, :p, :t, expression = Val{false})[2])
end

function geodesic_acceleration_test()
    metric_f = x -> schwarzschild_metric_iso_cart(x; rs = 1, c = 1)
    Symbolics.simplify(geodesic_acceleration_func( metric_f, 4 ))
end

function initial_ray_velocity(metric_func::Function, spatial_velocity, spatial_position)
    position = [0; spatial_position]

    met = metric_func(position)

    velocity = [0; spatial_velocity]
    velocity[1] = -velocity'*met*velocity
    velocity[1] /= met[1, 1]
    velocity[1] = sqrt( velocity[1] )
    return velocity
end

function initial_ray_position(spatial_position)
    [0; spatial_position]
end

function trace_ode_termination_cb(state,_,_)
    state[1]^2 - 10e5
end

function run_ode_trace(
        ode_func, metf,
        initial_spatial_velocity, initial_spatial_position,
        time_final
    )

    du0 = initial_ray_velocity(metf, initial_spatial_velocity, initial_spatial_position)
    u0 = initial_ray_position(initial_spatial_position)
    
    termination = ContinuousCallback(trace_ode_termination_cb, terminate!)

    prob = SecondOrderODEProblem(ode_func, du0, u0, (0, time_final))

    solve(prob, RK4(), callback = termination)
end

# function run_ode_traces(
#         ode_func, metf,
#         initial_spatial_velocities, initial_spatial_positions,
#         time_final
#     )
#     # du0 = initial_ray_velocity(metf, initial_spatial_velocity, initial_spatial_position)
#     du0s = map(initial_spatial_velocities, initial_spatial_positions) do v0, x0
#         initial_ray_velocity(metf, v0, x0)
#     end
#     u0s = map(initial_spatial_positions) do x0
#         initial_ray_position(x0)
#     end
#     termination = ContinuousCallback(trace_ode_termination_cb, terminate!)
#     prob = SecondOrderODEProblem(ode_func, du0, u0, (0, time_final))
#     solve(prob, RK4(), callback = termination)
# end

function run_ode_circle_tests()
    metric_f = x -> schwarzschild_metric_iso_cart(x; rs = 1, c = 1)
    ode_f = geodesic_acceleration_func(metric_f, 4)

    time_final = 50
    @time ThreadsX.map(range(0, 2*pi, length=1000)) do theta
        run_ode_trace(
                      ode_f, metric_f,
                      Float64[sincos(theta)...,0], Float64[5,0,0],
                      time_final
                     )
    end
end

function fibonacci_sphere(samples=1000; T=Float64)

    points = Vector{T}[]
    phi::T = pi * (sqrt(T(5)) - one(T))  # golden angle in radians

    for i in 0:samples-1
        y::T = 1 - (i / T(samples - 1)) * 2  # y goes from 1 to -1
        radius::T = sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = cos(theta) * radius
        z = sin(theta) * radius

        push!(points, [x, y, z])
    end

    return points
end

function run_ode_circle_sphere_test(n)
    metric_f = x -> schwarzschild_metric_iso_cart(x; rs = 1, c = 1)
    ode_f = geodesic_acceleration_func(metric_f, 4)

    initial_directions = fibonacci_sphere(n)

    time_final = 100
    @time traces = ThreadsX.map(initial_directions) do dir
        run_ode_trace(
                      ode_f, metric_f,
                      dir, Float64[0,2,0],
                      time_final
                     )
    end

    final_directions = map(traces) do trace
        normalize(trace.u[end][2:end÷2])
    end

    (initial_directions, final_directions)
end

function cart_to_sphere(x)
    r = norm(x)
    [
     atan(x[2], x[1]),
     atan(r, x[3])
    ]
end

end # module GRTracing
