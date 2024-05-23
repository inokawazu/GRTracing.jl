function metric_accel(metf, imetf, velocity::StaticVector{N}, position::StaticVector{N}) where {N}

    dmet = reshape(ForwardDiff.jacobian(metf, position), (N, N, N))

    imetp = imetf(position)
    ind = SVector{N}(1:N)

    rhs = map(ind) do i
        sum(
            velocity[k] * velocity[j] * (-dmet[i,j,k] + dmet[k,j,i]/2)
            for k in ind for j in ind
           )
    end

    map(ind) do i
        sum(imetp[i,j] * rhs[j] for j in ind)
    end
end
