function lagrangian(met::AbstractMatrix, velocity::AbstractVector)
    velocity'*met*velocity/2
end

function lagrangian(metf::Function, velocity, position)
    # dot(velocity, metf(position), velocity)/2
    lagrangian(metf(position), velocity)
end

function lagrangian(metf::Function, vp::StaticVector{N}) where N
    vind = SVector{N÷2}(1:N÷2)
    pind = SVector{N÷2}(N÷2+1:N)
    lagrangian(metf, vp[vind], vp[pind])
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

    lvivj \ (-lvixj*velocity + lxj)
end
