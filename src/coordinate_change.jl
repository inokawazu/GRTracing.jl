function cart_to_sphere(x)
    r = norm(x)
    [
     atan(x[2], x[1])
     atan(r, x[3])
    ]
end

function uv_to_spherical(pixel_coord;
        hor_angle_of_view = π/2,
        ver_angle_of_view = π/2,
    )
    u, v = pixel_coord
    θ = hor_angle_of_view * u - hor_angle_of_view/2
    ϕ = ver_angle_of_view * v + (pi - ver_angle_of_view)/2

    return θ, ϕ
end

function sphere_to_uv((the, phi))
    [
     the/2pi
     phi/pi
    ]
end
