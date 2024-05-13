function geodesic_acceleration_test()
    metric_f = x -> schwarzschild_metric_iso_cart(x; rs = 1, c = 1)
    Symbolics.simplify(geodesic_acceleration_func( metric_f, 4 ))
end

function run_ode_circle_test()
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
