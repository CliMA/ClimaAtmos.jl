# we are ignoring the volume reflectance for now
abstract type SurfaceAlbedoModel end

struct CouplerAlbedo <: SurfaceAlbedoModel end

"""
    struct ConstantAlbedo{FT} <: SurfaceAlbedoModel

A constant surface albedo model. The default value is `α = 0.38`.
It is used purely for idealized experiments.
"""
@kwdef struct ConstantAlbedo{FT} <: SurfaceAlbedoModel
    α::FT
end

"""
    struct RegressionFunctionAlbedo{FT} <: SurfaceAlbedoModel

A regression function-based surface albedo model of Jin et al. (2011, referred to as J11).
"""
struct RegressionFunctionAlbedo{FT, F <: Function} <: SurfaceAlbedoModel
    n::FT                           # relative refractive index of water and air (n = n_w/n_a) TODO: f(wavelength) for a spectrally dependent scheme
    n0::FT                          # refractive index of water for visible light
    p::SA.SVector{11, FT}           # regression coefficients of J11 for eq. 4
    q_clear::SA.SVector{4, FT}      # regression coefficients of J11 for eq. 5a
    q_cloud::SA.SVector{3, FT}      # regression coefficients of J11 for eq. 5b
    wave_slope::F                   # mean wave slope distribution width (Cox-Munk model, J11 Eq. 2)
end
function RegressionFunctionAlbedo{FT}(;
    n = FT(1.34),
    n0 = FT(1.34),
    p = SA.SVector{11, FT}(
        0.0152,
        -1.7873,
        6.8972,
        -8.5778,
        4.071,
        -7.6446,
        0.1643,
        -7.8409,
        -3.5639,
        -2.3588,
        10.0538,
    ),
    q_clear = SA.SVector{4, FT}(-0.1482, -0.012, 0.1608, -0.0244),
    q_cloud = SA.SVector{3, FT}(-0.1479, 0.1502, -0.016),
    wave_slope = u -> sqrt(FT(0.003) + FT(0.00512) * u),
) where {FT <: AbstractFloat}
    return RegressionFunctionAlbedo(n, n0, p, q_clear, q_cloud, wave_slope)
end

"""
    set_surface_albedo!(Y, p, t, α_model::ConstantAlbedo)

Set the surface albedo to a constant value.
"""
function set_surface_albedo!(Y, p, t, α_model::ConstantAlbedo{FT}) where {FT}

    (; direct_sw_surface_albedo, diffuse_sw_surface_albedo) =
        p.radiation.rrtmgp_model

    @. direct_sw_surface_albedo = α_model.α
    @. diffuse_sw_surface_albedo = α_model.α
end

"""
    set_surface_albedo!(Y, p, t, α_model::RegressionFunctionAlbedo{FT})

Wrapper to call the formulations of Jin et al. (2011), and set the direct and diffuse surface albedos of the ocean.
"""
function set_surface_albedo!(
    Y,
    p,
    t,
    α_model::RegressionFunctionAlbedo{FT},
) where {FT}

    (; direct_sw_surface_albedo, diffuse_sw_surface_albedo, cos_zenith) =
        p.radiation.rrtmgp_model

    λ = FT(0) # spectral wavelength (not used for now)
    μ = cos_zenith

    surface_albedo = p.scratch.temp_field_level
    f_direct = surface_albedo_direct(α_model)
    surface_albedo .=
        f_direct.(
            λ,
            Fields.array2field(μ, axes(surface_albedo)),
            norm.(Fields.level(Y.c.uₕ, 1)),
        )
    direct_sw_surface_albedo .= Fields.field2array(surface_albedo)'

    f_diffuse = surface_albedo_diffuse(α_model)
    surface_albedo .=
        f_diffuse.(
            λ,
            Fields.array2field(μ, axes(surface_albedo)),
            norm.(Fields.level(Y.c.uₕ, 1)),
        )
    diffuse_sw_surface_albedo .= Fields.field2array(surface_albedo)'
end

"""
    set_surface_albedo!(Y, p, t, ::CouplerAlbedo)

Tell the ClimaAtmos to skip setting the surface albedo, as it is handled by the coupler.

When running in a coupled simulation, set the surface albedo to 0.38 at the beginning of the simulation,
so the initial callback initialization doesn't lead to NaNs in the radiation model.
Subsequently, the surface albedo will be updated by the coupler.
"""
function set_surface_albedo!(Y, p, t, ::CouplerAlbedo)
    FT = eltype(Y)
    if FT(t) == 0
        # set initial insolation initial conditions
        !(p.atmos.insolation isa IdealizedInsolation) &&
            set_insolation_variables!(Y, p, t, p.atmos.insolation)
        # set surface albedo to 0.38
        @warn "Setting surface albedo to 0.38 at the beginning of the simulation"
        p.radiation.rrtmgp_model.direct_sw_surface_albedo .= FT(0.38)
        p.radiation.rrtmgp_model.diffuse_sw_surface_albedo .= FT(0.38)
    else
        nothing
    end
end

"""
    surface_albedo_direct(α_model::RegressionFunctionAlbedo{FT}) where {FT}

Calculate the direct surface albedo using the regression function of Jin et al. (2011).
"""
function surface_albedo_direct(α_model::RegressionFunctionAlbedo{FT}) where {FT}
    α_dir =
        (λ, cosθ, u) -> begin
            if cosθ <= 0
                return zero(FT)
            else
                # relative refractive index of water and air (n = n_w/n_a)
                n = α_model.n

                # refractive index of water for visible light
                n0 = α_model.n0

                # mean wave slope distribution width
                σ = α_model.wave_slope(u)

                # Fresnel reflectance (assuming equal contribution of the p-polorized and s-polarized components, and using the perfect dielectric medium approximation)
                sinθ(cosθ) = sqrt(1 - cosθ^2)
                rf_p(n, cosθ) =
                    (
                        (n^2 * cosθ - sqrt(n^2 - sinθ(cosθ)^2)) /
                        (n^2 * cosθ + sqrt(n^2 - sinθ(cosθ)^2))
                    )^2
                rf_s(n, cosθ) =
                    (
                        (cosθ - sqrt(n^2 - sinθ(cosθ)^2)) /
                        (cosθ + sqrt(n^2 - sinθ(cosθ)^2))
                    )^2
                rf(n, cosθ) = (rf_p(n, cosθ) + rf_s(n, cosθ)) / 2

                # regression coefficients
                p = α_model.p

                # the regression function (J11, eq. 4)
                f(cosθ, σ) =
                    (
                        p[1] +
                        p[2] * cosθ +
                        p[3] * cosθ^2 +
                        p[4] * cosθ^3 +
                        p[5] * σ +
                        p[6] * σ * cosθ
                    ) * exp(
                        p[7] +
                        p[8] * cosθ +
                        p[9] * cosθ^2 +
                        p[10] * σ +
                        p[11] * σ * cosθ,
                    )

                # return the albedo (J11, eq. 1)
                return min(
                    one(FT),
                    max(
                        zero(FT),
                        rf(n, cosθ) - rf(n, cosθ) / rf(n0, cosθ) * f(cosθ, σ),
                    ),
                )
            end
        end
    return α_dir
end

"""
    surface_albedo_diffuse!(α_model::RegressionFunctionAlbedo)

Calculate the diffuse surface albedo using the Jin et al. (2011) empirical formulation.

!!! note
    For now we assume that the cloud fraction is 0.0.
"""
function surface_albedo_diffuse(
    α_model::RegressionFunctionAlbedo{FT},
) where {FT}
    α_diff =
        (λ, cosθ, u) -> begin
            if cosθ <= 0
                return zero(FT)
            else
                cloud_fraction = 0 # TODO: connect this to the EDMF

                # clear sky (J11, eq. 5a)
                n = α_model.n
                σ = α_model.wave_slope(u)

                q_clear = α_model.q_clear
                α_clear =
                    q_clear[1] +
                    q_clear[2] * σ +
                    q_clear[3] * n +
                    q_clear[4] * n * σ

                # cloudy sky (J11, eq. 5b)
                q_cloud = α_model.q_cloud
                α_cloud = q_cloud[1] + q_cloud[2] * n + q_cloud[3] * n * σ

                # return the albedo
                return min(
                    one(FT),
                    max(
                        zero(FT),
                        cloud_fraction * α_cloud +
                        (1 - cloud_fraction) * α_clear,
                    ),
                )
            end
        end
    return α_diff
end
