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

    gf = build_function(accel, velocity, position, expression = Val{false})[2]
    return @eval $gf
end

function geodesic_acceleration_test()
    metric_f = x -> schwarzschild_metric_iso_cart(x; rs = 1, c = 1)
    Symbolics.simplify(geodesic_acceleration_func( metric_f, 4 ))
end

function make_ode_function(metric::Function, dim::Integer)
    let f = geodesic_acceleration_func(metric, dim)
        function (ddu, du, u, _, _)
            return f(ddu, du, u)
        end
    end
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

function run_ode_circle_tests()
    metric_f = x -> schwarzschild_metric_iso_cart(x; rs = 1, c = 1)
    ode_f = make_ode_function(metric_f, 4)

    time_final = 50
    @time ThreadsX.map(range(0, 2*pi, length=1000)) do theta
        run_ode_trace(
                      ode_f, metric_f,
                      Float64[sincos(theta)...,0], Float64[5,0,0],
                      time_final
                     )
    end
end

end # module GRTracing
