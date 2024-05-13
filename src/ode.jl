function trace_ode_termination_cb(vtcutoff=10e3)
    let vtcutoff=vtcutoff
        cb = (state,_,_) -> norm(state[1]) - vtcutoff
        ContinuousCallback(cb, terminate!)
    end
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

function run_ode_trace(
        ode_func, metf,
        initial_spatial_velocity, initial_spatial_position,
        time_final;
        termination_cb = trace_ode_termination_cb()
    )

    du0 = initial_ray_velocity(metf, initial_spatial_velocity, initial_spatial_position)
    u0 = initial_ray_position(initial_spatial_position)
    

    prob = SecondOrderODEProblem(ode_func, du0, u0, (zero(time_final), time_final))

    solve(prob, RK4(), callback = termination_cb)
end
