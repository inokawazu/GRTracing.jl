"""
    schwarzschild_metric_iso_cart((t,x,y,z); rs, c = 1)
using OrdinaryDiffEq: FiniteDiff
"""
function schwarzschild_metric_iso_cart((t,x,y,z); rs, c = 1)
    R = (sqrt∘sum)(x->x^2,(x,y,z))
    rm = (1 - rs/R)^2
    rp = (1 + rs/R)^2
    return SA[
     -rm/rp*c^2 0 0 0
     0 rp^2 0 0
     0 0 rp^2 0
     0 0 0 rp^2
    ]
end

function lagrangian(met::AbstractArray, velocity, position)
    1/2 * dot(velocity, met, velocity)
end

function lagrangian(met::AbstractArray, vp::StaticVector{N}) where N
    @views lagrangian(met, vp[1:N÷2], vp[N÷2+1:N])
end

function lagrangian(metf::Function, velocity, position)
    1/2 * dot(velocity, metf(position), velocity)
end

function lagrangian(metf::Function, vp::StaticVector{N}) where N
    @views lagrangian(metf, vp[1:N÷2], vp[N÷2+1:N])
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
    # lvivj aj + lvixj vj =  lxixj vj
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


"""
    soft_lump_metric_iso_cart((t,x,y,z); rs, c = 1)
"""
@inline function soft_lump_metric_iso_cart((t,x,y,z); rs = 1.0, c = 1.0)
    R = hypot(x,y,z)
    rp = (1 + rs/R)^2
    return SA[
     -c^2 0 0 0
     0 rp^2 0 0
     0 0 rp^2 0
     0 0 0 rp^2
    ]
end
