function init_bickley_jet_2d_plane(params)
    @unpack h₀, l, k, ϵ = params

    # density
    h_init(local_geometry) = h₀

    # velocity
    u_init(local_geometry) = begin
        @unpack x1, x2 = local_geometry.coordinates

        U₁ = cosh(x2)^(-2)

        # Ψ′ = exp(-(x2 + l / 10)^2 / 2l^2) * cos(k * x1) * cos(k * x2)
        # Vortical velocity fields (u₁′, u₂′) = (-∂²Ψ′, ∂¹Ψ′)
        gaussian = exp(-(x2 + l / 10)^2 / 2 * l^2)
        u₁′ = gaussian * (x2 + l / 10) / l^2 * cos(k * x1) * cos(k * x2)
        u₁′ += k * gaussian * cos(k * x1) * sin(k * x2)
        u₂′ = -k * gaussian * sin(k * x1) * cos(k * x2)
        u = Geometry.Covariant12Vector(
            Geometry.Cartesian12Vector(U₁ + ϵ * u₁′, ϵ * u₂′),
            local_geometry,
        )

        return u
    end

    # passive tracer
    c_init(local_geometry) = begin
        @unpack x2 = local_geometry.coordinates

        return sin(k * x2)
    end

    return (h = h_init, u = u_init, c = c_init)
end
