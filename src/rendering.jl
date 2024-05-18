struct LagrangianRender{F, G, T, V}
    lagf::F
    metf::G
    position::T
    cam_matrix::V
end

function LagrangianRender(metf::Function, camera_direction, cam_position)
    lagf = vp -> lagrangian(metf, vp)
    camera_direction = normalize(camera_direction)
    cam_matrix = [camera_direction nullspace(camera_direction*camera_direction')]
    cam_matrix = SMatrix{size(cam_matrix)...}(cam_matrix)
    return LagrangianRender(lagf, metf, cam_position, cam_matrix)
end


function trace_ode_termination_cb(vtcutoff=10e2)
    let vtcutoff=vtcutoff
        cb = (state,_,_) -> norm(state[1]) - vtcutoff
        ContinuousCallback(cb, terminate!)
    end
end

function render_pixel2heading(
        met_renderer::LagrangianRender, pixel_coord;
        time_final = 1000
    )

    (;position, cam_matrix, lagf, metf) = met_renderer

    (θ, ϕ) = uv_to_spherical(pixel_coord)

    x = sin(ϕ) * cos(θ)
    y = sin(ϕ) * sin(θ)
    z = cos(ϕ)

    initial_heading = cam_matrix * SA[x, y, z]

    u0  = initial_ray_position(position)
    du0 = initial_ray_velocity(vcat(0, initial_heading), SA[1.0, 0, 0, 0], metf(u0))

    function ode_func(du, u, _, _)
        lagrangian_accel(lagf, du, u)
    end

    prob = SecondOrderODEProblem(ode_func, du0, u0, (zero(time_final), time_final))
    sol = solve(prob, RK4(), save_on = false, callback=trace_ode_termination_cb())

    return sol.u[end].x[1]
end

function render_sky(mrend::LagrangianRender, view_size::Integer; sky_image = load_test_image("test_sky_1.png"))
    us = range(0.0, 1.0, length=view_size)
    vs = range(0.0, 1.0, length=view_size)

    out4vecs = ThreadsX.map(Iterators.product(us, vs)) do (u, v)
        render_pixel2heading(mrend, (u, v))
    end

    black = zero(eltype(sky_image))

    map(out4vecs) do out4vec
        if norm(out4vec[1]) > 100
            return black
        end

        (u, v) = (sphere_to_uv∘cart_to_sphere)(out4vec[2:end])
        
        i = clamp(ceil(Int, v * size(sky_image, 1)), 1, size(sky_image, 1))
        j = clamp(ceil(Int, u * size(sky_image, 2)), 1, size(sky_image, 2))
        sky_image[i, j]
    end
end

# struct MetricRenderer{F <: Function, G <: Function, T, V}
#     metric_function::F
#     ode_function::G
#     position::T
#     # camera_direction::T
#     cam_matrix::V
# end

# function MetricRenderer(metric_function::Function, position, camera_direction, dim = 4)
#     ode_function = geodesic_acceleration_func(metric_function, dim)
#     camera_direction = normalize(camera_direction)
#     cam_matrix = [camera_direction nullspace(camera_direction*camera_direction')]
#     return MetricRenderer(
#                           metric_function, ode_function,
#                           position, cam_matrix
#                          )
# end

# """
#     render_pixel2heading(
#         met_renderer::MetricRenderer, pixel_coord, position, camera_direction;
#         time_final = 300
#     )

# Render renders pixel with spacetime.

# pixel_coord -> sphere_point (initial heading) -> translated heading

# If the translated heading is not at infinity, the returned pixel is black.

# Else, it's mapped to a point on the celestial sphere -> pixel_coord.
# """
# function render_pixel2heading(
#         met_renderer::MetricRenderer, pixel_coord;
#         time_final = 300
#     )

#     (;position, cam_matrix) = met_renderer

#     # camera_direction = normalize(camera_direction)

#     (θ, ϕ) = uv_to_spherical(pixel_coord)

#     x = sin(ϕ) * cos(θ)
#     y = sin(ϕ) * sin(θ)
#     z = cos(ϕ)

#     initial_heading = cam_matrix * [x, y, z]

#     trace = run_ode_trace(met_renderer.ode_function, met_renderer.metric_function,
#                   initial_heading, position, time_final)

#     trace.u[end][1:end÷2]
# end
