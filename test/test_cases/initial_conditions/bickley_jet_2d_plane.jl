function init_bickley_jet_2d_plane(local_geometry, params)
    @unpack ρ₀, l, k, ϵ = params
    @unpack x1, x2 = local_geometry.coordinates

    # density
    ρ = ρ₀

    # velocity
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

    # passive tracer
    θ = sin(k * x2)

    return (ρ = ρ, u = u, ρθ = ρ * θ)
end
