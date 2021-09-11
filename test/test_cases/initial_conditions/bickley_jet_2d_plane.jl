function init_bickley_jet_2d_plane(params)
    @unpack h₀, l, k, ϵ = params

    # density
    h_init(local_geometry) = h₀

    # velocity
    u_init(local_geometry) = begin
        @unpack x, y = local_geometry.coordinates

        U₁ = cosh(y)^(-2)

        # Ψ′ = exp(-(y + l / 10)^2 / 2l^2) * cos(k * x) * cos(k * y)
        # Vortical velocity fields (u₁′, u₂′) = (-∂²Ψ′, ∂¹Ψ′)
        gaussian = exp(-(y + l / 10)^2 / 2 * l^2)
        u₁′ = gaussian * (y + l / 10) / l^2 * cos(k * x) * cos(k * y)
        u₁′ += k * gaussian * cos(k * x) * sin(k * y)
        u₂′ = -k * gaussian * sin(k * x) * cos(k * y)
        u = Geometry.Covariant12Vector(
            Geometry.Cartesian12Vector(U₁ + ϵ * u₁′, ϵ * u₂′),
            local_geometry,
        )

        return u
    end

    # passive tracer
    c_init(local_geometry) = begin
        @unpack y = local_geometry.coordinates

        return sin(k * y)
    end

    return (h = h_init, u = u_init, c = c_init)
end
