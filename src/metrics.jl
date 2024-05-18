"""
    schwarzschild_metric_iso_cart((t,x,y,z); rs, c = 1)
using OrdinaryDiffEq: FiniteDiff
"""
function schwarzschild_metric_iso_cart((t,x,y,z); rs, c = 1)
    # R = (sqrt∘sum)(x->x^2, (x,y,z))
    R = hypot(x,y,z)
    rm = (1 - rs/R)^2
    rp = (1 + rs/R)^2
    return SA[
     -rm/rp*c^2 0 0 0
     0 rp^2 0 0
     0 0 rp^2 0
     0 0 0 rp^2
    ]
end

"""
    soft_lump_metric_iso_cart((t,x,y,z); rs, c = 1)
"""
function soft_lump_metric_iso_cart((t,x,y,z); rs, c = 1)
    R = hypot(x,y,z)
    # rp = 1 + rs/(R + rs)
    rp = (1 + rs/(R+rs))^2
    return SA[
     -c^2 0 0 0
     0 rp^2 0 0
     0 0 rp^2 0
     0 0 0 rp^2
    ]
end

"""
    minkowski_cart((t,x,y,z); c = 1)

Flat spacetime metric 
"""
function minkowski_cart((t,x,y,z); c = 1)
    return SA[
     -c^2 0 0 0
     0 1 0 0
     0 0 1 0
     0 0 0 1
    ]
end

"""
https://physics.stackexchange.com/a/568769

`re` ≡ *Radius* of *Ergosphere*

TODO: Bounds on `a` w.r.t `re`
"""
function kerr_metric_pseudo_cart((t,x,y,z); re, a, c = 1)
    m = re^2 / 2
    r = sqrt(-a^2 + x^2 + y^2 + z^2 + sqrt(4*a^2*z^2 + (-a^2 + x^2 + y^2 + z^2)^2 ))/sqrt(2)
    rot_fact = 2m*r^3/(r^4 + a^2*z^2)
    rot_vec = SA[
                 1
                 (r*x + a*y)/(a^2+r^2)
                 (r*y - a*x)/(a^2+r^2)
                 z/r
                ]
    flat_part = SA[
                   -c^2 0 0 0
                   0 1 0 0
                   0 0 1 0
                   0 0 0 1
                  ]
    return flat_part + rot_fact * rot_vec * rot_vec'
end
