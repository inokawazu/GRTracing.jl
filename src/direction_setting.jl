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
