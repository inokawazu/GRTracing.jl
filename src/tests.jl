const TEST_IMAGE_DIR = joinpath(@__DIR__, "..", "assets", "test_images")

function load_test_image(filename)
    load(joinpath(TEST_IMAGE_DIR, filename))
end

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

function test_lagrangian()
    metf = x -> soft_lump_metric_iso_cart(x; rs = 1.0)
    position = SA{Float64}[3,3,3,3]
    velocity = SA{Float64}[1,2,3,4]
    metp = metf(position)

    @time lagp = lagrangian(metf, velocity, position)
    @show lagp

    myvp = vcat(velocity, position)
    lagf = vp -> lagrangian(metf, vp)

    # lagfp = lagf(myvp)
    @time laghp = ForwardDiff.hessian(lagf, myvp)
    @show laghp

    display(metp)
    display(laghp)

    lvivj = @view laghp[1:end÷2, 1:end÷2] 
    lvixj = @view laghp[1:end÷2, end÷2+1:end] 
    lxixj = @view laghp[end÷2+1:end, end÷2+1:end] 
    accp = lvivj\(-lvixj*velocity + lxixj*velocity)
end


function test_metric_accel()
    position = SA{Float64}[3,3,3,3]
    velocity = SA{Float64}[1,2,3,4]
    # acceleration = similar(velocity)

    metf = x -> soft_lump_metric_iso_cart(x; rs = 1.0)
    imetf = inv∘metf

    @time metric_accel(metf, imetf, velocity, position)
    @time metric_accel(metf, imetf, velocity, position)

    nruns = 100_000
    @time (foreach(1:nruns) do _
         metric_accel(metf, imetf, velocity, position)
     end)

    # nruns = 1920*1080
    # @time (ThreadsX.foreach(1:nruns) do _
    #      lagrangian_accel!(lagf, acceleration, velocity, position)
    #  end)
end

function test_lagrangian_accel()
    position = SA{Float64}[3,3,3,3]
    velocity = SA{Float64}[1,2,3,4]
    acceleration = similar(velocity)

    metf = x -> soft_lump_metric_iso_cart(x; rs = 1.0)
    lagf = vp -> lagrangian(metf, vp)
    # lagf = let metp = metf(position)
    #     vp -> lagrangian(metp, vp)
    # end

    # @time lagrangian_accel!(lagf, acceleration, velocity, position)
    # @time lagrangian_accel!(lagf, acceleration, velocity, position)
    @time lagrangian_accel(lagf, velocity, position)
    @time lagrangian_accel(lagf, velocity, position)

    nruns = 100_000
    @time (foreach(1:nruns) do _
         lagrangian_accel(lagf, velocity, position)
     end)

    # nruns = 1920*1080
    # @time (ThreadsX.foreach(1:nruns) do _
    #      lagrangian_accel!(lagf, acceleration, velocity, position)
    #  end)
end

function test_lag_render()
    cam_position = SA{Float64}[-20,0,0]
    cam_direction = normalize(SA{Float64}[1,0,0])
    uv = (0.5, 0.5)

    metf = x -> soft_lump_metric_iso_cart(x; c = 1.0, rs = 1.0)

    renderer = LagrangianRender(metf, cam_direction, cam_position)

    @time render_pixel2heading(renderer, uv)
    @time render_pixel2heading(renderer, uv)
end

function test_bulk_lag_render(view_size = 30)
    cam_position = SA{Float64}[-20,0,0]
    cam_direction = normalize(SA{Float64}[1,0,0])
    # uv = (0.5, 0.5)

    metf = x -> soft_lump_metric_iso_cart(x; c = 1.0, rs = 1.0)

    renderer = LagrangianRender(metf, cam_direction, cam_position)

    us = range(0.0, 1.0, length=view_size)
    vs = range(0.0, 1.0, length=view_size)
    uvs = Iterators.product(us, vs)

    # out4vecs = ThreadsX.map(Iterators.product(us, vs)) do (u, v)
    #     render_pixel2heading(mrend, (u, v))
    # end

    @time render_pixels2headings(renderer, collect(uvs))
    # @time render_pixels2headings(renderer, uvs)
end

function test_lag_metric_sky(view_size::Integer, mfunc, view_image_name; distance = 200)
    sky_image = load_test_image("test_sky_1.png")

    mrend = LagrangianRender(mfunc, SA{Float64}[0,1,0], SA{Float64}[0, -distance, 0])
    view_image = render_sky(mrend, view_size, sky_image=sky_image)
    save("$view_image_name.png", view_image)

    @info "Finished $view_image_name at view size of $view_size."
end

function test_lag_render_sky(view_size::Integer; distance = 200)
    sky_image = load_test_image("test_sky_1.png")
    mfuncs = [
              x -> schwarzschild_metric_iso_cart(x; rs = 1.0, c = 1.0)
              x -> soft_lump_metric_iso_cart(x; rs = 1.0, c = 1.0)
              x -> minkowski_cart(x; c = 1.0)
              x -> kerr_metric_pseudo_cart(x; re = 1.0, a=0.5, c = 1.0)
             ]

    for (i, mfunc) in enumerate(mfuncs)
        mrend = LagrangianRender(mfunc, SA{Float64}[0,1,0], SA{Float64}[0, -distance, 0])
        @time view_image = render_sky(mrend, view_size, sky_image=sky_image)
        save("test_view_image_$(i)_$(view_size).png", view_image)
        @info "Finished $i at view size of $view_size."
    end
end

# function render_test()
#     mfunc = x -> schwarzschild_metric_iso_cart(x; rs = 1, c = 1)
#     mrend = MetricRenderer(mfunc, Float64[-20, 0, 0], Float64[1,0,0])
#     out4vecs = map(Iterators.product(0.0:0.1:1.0, 0.0:0.1:1.0)) do (u, v)
#         render_pixel2heading(mrend, (u, v))
#     end

#     first.(out4vecs)
# end

# # Timsings
# # 16 Threads ~ 295a834 ~ 0.89s
# # 16 Threads ~ 17fc4cf ~ 0.85s
# function render_test_100x100_picture()
#     render_test_picture(100)
# end

# function render_test_picture(view_size::Integer)
#     mfunc = x -> schwarzschild_metric_iso_cart(x; rs = 1, c = 1)
#     mrend = MetricRenderer(mfunc, Float64[0, -100, 0], Float64[0,1,0])
#     us = range(0.0, 1.0, length=view_size)
#     vs = range(0.0, 1.0, length=view_size)


#     @time out4vecs = ThreadsX.map(Iterators.product(us, vs)) do (u, v)
#         render_pixel2heading(mrend, (u, v))
#     end

#     test_image = load_test_image("test_sky_1.png")
#     black = zero(eltype(test_image))

#     view_image = map(out4vecs) do out4vec
#         if norm(out4vec[1]) > 100
#             return black
#         end

#         (u, v) = (sphere_to_uv∘cart_to_sphere)(out4vec[2:end])
        
#         i = clamp(ceil(Int, v * size(test_image, 1)), 1, size(test_image, 1))
#         j = clamp(ceil(Int, u * size(test_image, 2)), 1, size(test_image, 2))
#         test_image[i, j]
#     end

#     @show size(view_image)
#     save("test_view_image_$(view_size).png", view_image)
# end

# function render_sky(mrend::MetricRenderer, view_size::Integer, sky_image)
#     us = range(0.0, 1.0, length=view_size)
#     vs = range(0.0, 1.0, length=view_size)

#     out4vecs = ThreadsX.map(Iterators.product(us, vs)) do (u, v)
#         render_pixel2heading(mrend, (u, v))
#     end

#     black = zero(eltype(sky_image))

#     map(out4vecs) do out4vec
#         if norm(out4vec[1]) > 100
#             return black
#         end

#         (u, v) = (sphere_to_uv∘cart_to_sphere)(out4vec[2:end])
        
#         i = clamp(ceil(Int, v * size(sky_image, 1)), 1, size(sky_image, 1))
#         j = clamp(ceil(Int, u * size(sky_image, 2)), 1, size(sky_image, 2))
#         sky_image[i, j]
#     end
# end

# function test_render_sky(view_size::Integer)
#     sky_image = load_test_image("test_sky_1.png")
#     mfuncs = [
#               x -> schwarzschild_metric_iso_cart(x; rs = 1, c = 1)
#               x -> soft_lump_metric_iso_cart(x; rs = 1, c = 1)
#              ]


#     for (i, mfunc) in enumerate(mfuncs)
#         mrend = MetricRenderer(mfunc, Float64[0, -100, 0], Float64[0,1,0])
#         view_image = render_sky(mrend, view_size, sky_image)
#         save("test_view_image_$i.png", view_image)
#         @info "Finished $i"
#     end
# end
