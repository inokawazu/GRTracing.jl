module GRTracing

using ForwardDiff, LinearAlgebra, OrdinaryDiffEq, StaticArrays

struct Metric{F <: Function, G}
    components::F
    size::G
end

function Base.size(m::Metric)
    m.size
end

function (m::Metric)(position)
    m.components(position)
end

function Base.inv(m::Metric)
    Metric(
           ComposedFunction(inv, m.components),
           m.size
          )
end

"""
    schwarzschild_metric_function((t,x,y,z); rs, c = 1)
"""
function schwarzschild_metric_function((t,x,y,z); rs, c = 1)
    R = hypot(x,y,z)
    rm = (1 - rs/R)^2
    rp = (1 + rs/R)^2
    @SMatrix [
     -rm/rp*c^2 0 0 0
     0 rp^2 0 0
     0 0 rp^2 0
     0 0 0 rp^2
    ]
end

function schwarzschild_metric(; c = 1, rs = 1)
    Metric( 
           v -> schwarzschild_metric_function(v, rs=rs,c=c),
           (4, 4)
          )
end

function diff_metric(metric::Metric, position)
    out = ForwardDiff.jacobian(metric, position)
    return reshape(out, (size(metric)..., :))
end

"""
    geodesic_acceleration(metric, velocity, position)

    Gives the acceleration according to GR
"""
function geodesic_acceleration(metric::Metric, velocity, position)
    dmetric = diff_metric(metric, position)
    inv_metric = inv(metric)(position)

    [
     sum(
         -1/2*inv_metric[i, l] * (dmetric[n,l,m] + dmetric[m,l,n] - dmetric[m,n,l]) * velocity[n] * velocity[m]
         for m in 1:length(velocity), 
             n in 1:length(velocity), 
             l in 1:length(velocity)
        )
     for i in 1:length(velocity)
    ]
end

function initial_ray_velocity(metric::Metric, spatial_velocity, spatial_position)
    position = [0; spatial_position]

    met = metric(position)

    velocity = [0; spatial_velocity]
    velocity[1] = -velocity'*met*velocity
    velocity[1] /= met[1, 1]
    velocity[1] = sqrt( velocity[1] )
    return velocity
end

function initial_ray_position(spatial_position)
    [0; spatial_position]
end

function euler_step(metric::Metric, velocity, position, dt)
    new_velocity = dt * geodesic_acceleration(metric, velocity, position) + velocity
    new_position = dt * velocity  + position
    return (new_velocity, new_position)
end

function run_euler_trace(
        metric::Metric,
        initial_spatial_velocity, initial_spatial_position;
        time_final, dt
    )
    velocity = initial_ray_velocity(metric, 
                                    initial_spatial_velocity, 
                                    initial_spatial_position) 
    position = initial_ray_position(initial_spatial_position)
    for _ in range(0, time_final, step = dt)
        (velocity, position) = euler_step(metric, velocity, position, dt)
    end
    (velocity, position)
end

function make_ode_function(metric::Metric)
    function ode_function(u′, u, _, _)
        geodesic_acceleration(metric, u′, u)
    end
end

function run_ode_trace(
        metric::Metric,
        initial_spatial_velocity, initial_spatial_position,
        time_final
    )

    du0 = initial_ray_velocity(metric, initial_spatial_velocity, initial_spatial_position)
    u0 = initial_ray_position(initial_spatial_position)
    f = make_ode_function(metric)
    

    function cb(state,_,_)
        du, u = state[1:end÷2], state[end÷2+1:end]
        abs(det(metric(u))) < 10e-5 || dot(du, du) < 10e-5
    end

    termination = DiscreteCallback(cb, terminate!)

    prob = SecondOrderODEProblem(f, du0, u0, (0, time_final))

    solve(prob, RK4(), callback = termination)
end

# set routines

function run_ode_trace_test()
    metric = schwarzschild_metric(rs = 1)
    time_final = 100

    run_ode_trace(
                  metric,
                  Float64[1,-1,0], Float64[5,0,0],
                  time_final
                 )
end

function run_ode_circle_tests()
    metric = schwarzschild_metric(rs = 1)
    time_final = 200

    map(range(0, 2*pi, length=100)) do theta
        run_ode_trace(
                      metric,
                      Float64[sincos(theta)...,0], Float64[5,0,0],
                      time_final
                     )
    end
end

end # module GRTracing
