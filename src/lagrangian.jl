function lagrangian(met::AbstractMatrix, velocity::AbstractVector)
    velocity'*met*velocity/2
end

function lagrangian(metf::Function, velocity, position)
    lagrangian(metf(position), velocity)
end

function lagrangian(metf::Function, vp::StaticVector{N}) where N
    vind = SVector{N÷2}(1:N÷2)
    pind = SVector{N÷2}(N÷2+1:N)
    lagrangian(metf, vp[vind], vp[pind])
end

function lagrangian(metf::Function, vp::AbstractVector)
    lagrangian(metf, vp[1:end÷2], vp[end÷2+1:end])
end

function lagrangian!(metf::Function, lagrangians::AbstractVector, velocities::AbstractMatrix, positions::AbstractMatrix)
    map!(lagrangians, eachcol(velocities), eachcol(positions)) do velocity, position
        lagrangian(metf, velocity, position)
    end
end

function lagrangian!(metf::Function, lagrangians::AbstractVector, vps::AbstractVector)
    @views begin
        velocities = vps[1:end÷2, :]
        positions = vps[end÷2+1:end, :]
    end
    lagrangian!(metf, lagrangians, velocities, positions)
end

# Accelerations

# function lagrangian_accel(lagf, velocity::AbstractMatrix, position::AbstractMatrix)
function lagrangian_accel!(
        lagf, avs::AbstractMatrix, vps::AbstractMatrix;
        laghps = similar(avs, (size(avs, 1), size(avs, 1), size(avs, 2))),
        laggps = similar(avs, (size(avs, 1), size(avs, 2)))
    )

    foreach(eachslice(laghps, dims=3), eachcol(vps)) do laghp, vp
        ForwardDiff.hessian!(laghp, lagf, vp)
    end

    foreach(eachcol(laggps), eachcol(vps)) do laggp, vp
        ForwardDiff.gradient!(laggp, lagf, vp)
    end

    @views begin
        lvivjs = laghps[1:end÷2, 1:end÷2, :]
        lvixjs = laghps[1:end÷2, end÷2+1:end, :] 
        lxjs = laggps[end÷2+1:end, :]
        velocities = vps[1:end÷2, :]
        accels = avs[1:end÷2, :]
    end


    # foreach(
    #     eachslice(accels, dims=2),
    #     eachslice(lvivjs, dims=3),
    #     eachslice(lvixjs, dims=3), 
    #     eachslice(lxjs, dims=2), 
    #     eachslice(velocities, dims=2)) do accel, lvivj, lvixj, lxj, velocity
    #     # lvivj \ (-lvixj*velocity + lxj)
    #     accel .= lxj
    #     mul!(accel, lvixj, velocity, -1, 1)
    #     ldiv!(Diagonal(lvivj), accel)
    # end

    # avs[end÷2+1:end, :] .= velocities
    
    # (stack∘Iterators.map)(eachslice(lvivjs, dims=3),
    #     eachslice(lvixjs, dims=3), 
    #     eachslice(lxjs, dims=2), 
    #     eachslice(velocities, dims=2)) do lvivj, lvixj, lxj, velocity
    #     lvivj \ (-lvixj*velocity + lxj)
    # end

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
