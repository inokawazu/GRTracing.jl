"""
    schwarzschild_metric_iso_cart((t,x,y,z); rs, c = 1)
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
