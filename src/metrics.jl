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

function lagrangian(met::AbstractArray, velocity::AbstractVector)
    velocity'*met*velocity/2
end

# function lagrangian(met::AbstractArray, vp::StaticVector{N}) where N
#     vind = SVector{N÷2}(1:N÷2)
#     pind = SVector{N÷2}(N÷2+1:N)
#     lagrangian(met, vp[vind], vp[pind])
# end

function lagrangian(metf::Function, velocity, position)
    # dot(velocity, metf(position), velocity)/2
    lagrangian(metf(position), velocity)
end

function lagrangian(metf::Function, vp::StaticVector{N}) where N
    vind = SVector{N÷2}(1:N÷2)
    pind = SVector{N÷2}(N÷2+1:N)
    lagrangian(metf, vp[vind], vp[pind])
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

function lagrangian_accel!(lagf::T, accel::StaticVector{N}, velocity::StaticVector{N, VT}, position::StaticVector{N}) where {T, N, VT}
    vp = vcat(velocity, position)
    M = 2N

    laghp = similar(velocity, (M, M))
    ForwardDiff.hessian!(laghp, lagf, vp)

    laggp = similar(velocity, M)
    ForwardDiff.gradient!(laggp, lagf, vp)

    lvivj = @view laghp[1:M÷2, 1:M÷2]
    lvixj = @view laghp[1:M÷2, M÷2+1:M] 
    lxj = @view laggp[M÷2+1:M]

    accel .= lxj
    mul!(accel, lvixj, velocity, -1, 1)
    accel .= lvivj \ accel

    nothing
end

function lagrangian_accel(lagf::T, velocity::StaticVector{N, VT}, position::StaticVector{N}) where {T, N, VT}
    vp = vcat(velocity, position)

    laghp = ForwardDiff.hessian(lagf, vp)
    laggp = ForwardDiff.gradient(lagf, vp)

    vind = SVector{N}(1:N)
    pind = SVector{N}(N+1:2N)

    lvivj = laghp[vind, vind]
    lvixj = laghp[vind, pind] 
    lxj = laggp[pind]

    # @show laghp
    # @show laghp
    # display(laghp)
    # @show lvivj
    # @show det(lvivj)

    lvivj \ (-lvixj*velocity + lxj)
end

function test_lagrangian_accel()
    position = SA{Float64}[3,3,3,3]
    velocity = SA{Float64}[1,2,3,4]
    acceleration = similar(velocity)

    metf = x -> soft_lump_metric_iso_cart(x; rs = 1.0)
    # lagf = vp -> lagrangian(metf, vp)
    lagf = let metp = metf(position)
        vp -> lagrangian(metp, vp)
    end

    vp = vcat(velocity, position)
    @time lagf(vp);
    @time lagf(vp);

    @time lagrangian_accel!(lagf, acceleration, velocity, position)
    @time lagrangian_accel!(lagf, acceleration, velocity, position)

    nruns = 100_000
    @time (foreach(1:nruns) do _
         lagrangian_accel!(lagf, acceleration, velocity, position)
     end)

    nruns = 1920*1080
    @time (ThreadsX.foreach(1:nruns) do _
         lagrangian_accel!(lagf, acceleration, velocity, position)
     end)
end
