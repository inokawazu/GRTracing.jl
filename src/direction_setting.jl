function flat_initial_ray_velocity(svec)
    vcat(1, normalize(svec))
end

function initial_ray_velocity(svec, tvec, met)
    a = dot(svec, met, svec)
    b = dot(svec, met, tvec)
    c = dot(tvec, met, tvec)
    # TODO: account for other root
    scale = max((-b + sqrt(b^2 - a*c))/a, (-b - sqrt(b^2 - a*c))/a)
    return tvec + scale * svec
end

function flat_initial_ray_position(svec, tinit = 0)
    vcat(tinit, svec)
end

function initial_ray_position(args...)
    flat_initial_ray_position(args...)
end

function fibonacci_sphere(samples=1000; T=Float64, VT = Vector)

    points = VT{T}[]
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

function pixel_coord_to_heading(pixel_coord; cam_matrix)
    (θ, ϕ) = uv_to_spherical(pixel_coord)

    x = sin(ϕ) * cos(θ)
    y = sin(ϕ) * sin(θ)
    z = cos(ϕ)

    cam_matrix * SA[x, y, z]
end
