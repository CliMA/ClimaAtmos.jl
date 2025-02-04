#####
##### EDMF closures (nonhydrostatic pressure drag and mixing length)
#####

import StaticArrays as SA
import Thermodynamics.Parameters as TDP
import ClimaCore.Geometry as Geometry
import ClimaCore.Fields as Fields
using Flux

Base.broadcastable(x::AbstractMixingLengthModel) = tuple(x)
Base.broadcastable(x::AbstractPertPressureModel) = tuple(x)

"""
    Return draft area given ρa and ρ
"""
function draft_area(ρa, ρ)
    return ρa / ρ
end

"""
    Return buoyancy on cell centers.
"""
function ᶜphysical_buoyancy(thermo_params, ᶜρ_ref, ᶜρ)
    # TODO - replace by ᶜgradᵥᶠΦ when we move to deep atmosphere
    g = TDP.grav(thermo_params)
    return (ᶜρ_ref - ᶜρ) / ᶜρ * g
end
"""
    Return buoyancy on cell faces.
"""
function ᶠbuoyancy(ᶠρ_ref, ᶠρ, ᶠgradᵥ_ᶜΦ)
    return (ᶠρ_ref - ᶠρ) / ᶠρ * ᶠgradᵥ_ᶜΦ
end

"""
    Return surface flux of TKE, a C3 vector used by ClimaAtmos operator boundary conditions
"""
function surface_flux_tke(
    turbconv_params,
    ρ_int,
    u_int,
    ustar,
    interior_local_geometry,
    surface_local_geometry,
)
    c_d = CAP.tke_diss_coeff(turbconv_params)
    c_m = CAP.tke_ed_coeff(turbconv_params)
    k_star² = CAP.tke_surf_scale(turbconv_params)
    speed = Geometry._norm(
        CA.CT12(u_int, interior_local_geometry),
        interior_local_geometry,
    )
    c3_unit = C3(unit_basis_vector_data(C3, surface_local_geometry))
    return ρ_int * (1 - c_d * c_m * k_star²^2) * ustar^2 * speed * c3_unit
end

"""
   Return the nonhydrostatic pressure drag for updrafts [m/s2 * m], following He et al. (2022)

   Inputs (everything defined on cell faces):
   - params - set with model parameters
   - ᶠlg - local geometry (needed to compute the norm inside a local function)
   - ᶠbuoyʲ - covariant3 or contravariant3 updraft buoyancy
   - ᶠu3ʲ, ᶠu3⁰ - covariant3 or contravariant3 velocity for updraft and environment.
                  covariant3 velocity is used in prognostic edmf, and contravariant3
                  velocity is used in diagnostic edmf.
   - updraft top height
"""
function ᶠupdraft_nh_pressure(params, ᶠlg, ᶠbuoyʲ, ᶠu3ʲ, ᶠu3⁰, plume_height, ᶠz, ᶠbuoy⁰, ᶠtke⁰, ᶠaʲ, nh_pressure_type::PhysicalPertPressureModel)
    turbconv_params = CAP.turbconv_params(params)
    # factor multiplier for pressure buoyancy terms (effective buoyancy is (1-α_b))
    α_b = CAP.pressure_normalmode_buoy_coeff1(turbconv_params)
    # factor multiplier for pressure drag
    α_d = CAP.pressure_normalmode_drag_coeff(turbconv_params)

    # Independence of aspect ratio hardcoded: α₂_asp_ratio² = FT(0)

    H_up_min = CAP.min_updraft_top(turbconv_params)

    # We also used to have advection term here: α_a * w_up * div_w_up
    return α_b * ᶠbuoyʲ +
           α_d * (ᶠu3ʲ - ᶠu3⁰) * CC.Geometry._norm(ᶠu3ʲ - ᶠu3⁰, ᶠlg) /
           max(plume_height, H_up_min)
end

"""
   Return the data-driven nonhydrostatic pressure drag for updrafts [m/s2 * m]

   Inputs (everything defined on cell faces):
   - params - set with model parameters
   - ᶠlg - local geometry (needed to compute the norm inside a local function)
   - ᶠbuoyʲ - covariant3 or contravariant3 updraft buoyancy
   - ᶠu3ʲ, ᶠu3⁰ - covariant3 or contravariant3 velocity for updraft and environment.
                  covariant3 velocity is used in prognostic edmf, and contravariant3
                  velocity is used in diagnostic edmf.
   - updraft top height
"""
function ᶠupdraft_nh_pressure(params, ᶠlg, ᶠbuoyʲ, ᶠu3ʲ, ᶠu3⁰, plume_height, ᶠz, ᶠbuoy⁰, ᶠtke⁰, ᶠaʲ, w_up, w_en, nh_pressure_type::LinearPertPressureModel)
    turbconv_params = CAP.turbconv_params(params)

    param_vec = CAP.pressure_normalmode_param_vec(params)
    H_up_min = CAP.min_updraft_top(turbconv_params)
    FT = eltype(params)

    Π₁ = ᶠz * ᶠbuoyʲ[1] / ((w_up .- w_en)^2 + eps(FT)) / 100

    Π₂ = max(ᶠtke⁰, 0) / ((w_up - w_en)^2 + eps(FT)) / 2

    Π₃ = sqrt(ᶠaʲ)

    # buoyancy coefficient
    α_b = param_vec[1] * Π₁^param_vec[7] +
          param_vec[2] * Π₂^param_vec[8] +
          param_vec[3] * Π₃^param_vec[9] +
          param_vec[13]

    # drag coefficient
    α_d = param_vec[4] * Π₁^param_vec[10] +
          param_vec[5] * Π₂^param_vec[11] +
          param_vec[6] * Π₃^param_vec[12] +
          param_vec[14]

    return α_b * ᶠbuoyʲ +
        α_d * (ᶠu3ʲ - ᶠu3⁰) * CC.Geometry._norm(ᶠu3ʲ - ᶠu3⁰, ᶠlg) /
        max(plume_height, H_up_min)
end



edmfx_nh_pressure_tendency!(Yₜ, Y, p, t, turbconv_model) = nothing
function edmfx_nh_pressure_tendency!(
    Yₜ,
    Y,
    p,
    t,
    turbconv_model::PrognosticEDMFX,
)

    n = n_mass_flux_subdomains(turbconv_model)
    (; params) = p
    (; ᶠgradᵥ_ᶜΦ) = p.core
    (; ᶜtke⁰, ᶜρʲs, ᶠnh_pressure₃ʲs, ᶠu₃⁰, ᶜuʲs,) = p.precomputed
    ᶠlg = Fields.local_geometry_field(Y.f)
    ᶠz = Fields.coordinate_field(Y.f).z

    scale_height = CAP.R_d(params) * CAP.T_surf_ref(params) / CAP.grav(params)
    FT = eltype(params)

    for j in 1:n
        if p.atmos.edmfx_model.nh_pressure isa Val{true}
            @. ᶠnh_pressure₃ʲs.:($$j) = ᶠupdraft_nh_pressure(
                params,
                ᶠlg,
                ᶠbuoyancy(ᶠinterp(Y.c.ρ), ᶠinterp(ᶜρʲs.:($$j)), ᶠgradᵥ_ᶜΦ),
                Y.f.sgsʲs.:($$j).u₃,
                ᶠu₃⁰,
                scale_height,
                ᶠz,
                FT(0),
                max(ᶠinterp(ᶜtke⁰), 0),
                ᶠinterp(draft_area(Y.c.sgsʲs.:($$j).ρa, ᶜρʲs.:($$j))),
                get_physical_w(Y.f.sgsʲs.:($$j).u₃, ᶠlg),
                get_physical_w(ᶠu₃⁰, ᶠlg),
                p.atmos.nh_pressure_model,
            )
            @. Yₜ.f.sgsʲs.:($$j).u₃ -= ᶠnh_pressure₃ʲs.:($$j)
        else
            @. ᶠnh_pressure₃ʲs.:($$j) = C3(0)
        end
    end
end

# lambert_2_over_e(::Type{FT}) where {FT} = FT(LambertW.lambertw(FT(2) / FT(MathConstants.e)))
lambert_2_over_e(::Type{FT}) where {FT} = FT(0.46305551336554884) # since we can evaluate

function lamb_smooth_minimum(
    l::SA.SVector,
    lower_bound::FT,
    upper_bound,
) where {FT}
    x_min = minimum(l)
    λ_0 = max(x_min * lower_bound / lambert_2_over_e(FT), upper_bound)

    num = sum(l_i -> l_i * exp(-(l_i - x_min) / λ_0), l)
    den = sum(l_i -> exp(-(l_i - x_min) / λ_0), l)
    smin = num / den
    return smin
end

function mixing_length(
    params,
    ustar::FT,
    ᶜz::FT,
    z_sfc::FT,
    ᶜdz::FT,
    sfc_tke::FT,
    ᶜlinear_buoygrad::FT,
    ᶜtke::FT,
    obukhov_length::FT,
    ᶜstrain_rate_norm::FT,
    ᶜPr::FT,
    ᶜtke_exch::FT,
    ᶜwʲ::FT,
    ᶜw⁰::FT,
    ::NeuralNetworkMixingLengthModel,
) where {FT}

    param_vec = CAP.mixing_length_param_vec(params)

    l_z = ᶜz - z_sfc
    # X_vars = ["mix_len_pi1", "mix_len_pi2", "mix_len_pi3", "bgrad", "strain", "tke", "z_obu", "res_obu"]
    X_1 = ᶜstrain_rate_norm / (ᶜlinear_buoygrad + eps(FT)) # mix len pi 1
    X_2 = ᶜtke / (ᶜlinear_buoygrad * ᶜz^2) # mix len pi 2
    X_3 = ᶜtke / (((ᶜwʲ - ᶜw⁰)^2 + eps(FT))) # mix len pi 3
    X_4 = ᶜlinear_buoygrad #bgrad
    X_5 = ᶜstrain_rate_norm #strain
    X_6 = ᶜtke #tke
    X_7 = ᶜz / (obukhov_length + eps(FT)) # z_obu
    X_8 = ᶜdz / (obukhov_length + eps(FT)) # res_obu


    # clip to avoid blowup 
    X_1 = clamp(X_1, -100.0, 100.0)
    X_2 = clamp(X_2, -1e4, 1e4)
    X_3 = clamp(X_3, -100.0, 100.0)
    X_4 = clamp(X_4, -0.02, 0.02)
    X_5 = clamp(X_5, -1e-3, 1e-3)
    X_7 = clamp(X_7, -3e4, 3e4)
    X_8 = clamp(X_8, -2500, 2500)


    means = Dict{Symbol, FT}(
        :X_1 => FT(0.08152589946985245),    # mix_len_pi1
        :X_2 => FT(-9.220643997192383),     # mix_len_pi2
        :X_3 => FT(1.456505298614502),      # mix_len_pi3
        :X_4 => FT(4.150567838223651e-05),  # bgrad
        :X_5 => FT(1.2203592632431537e-05), # strain
        :X_6 => FT(0.1851629614830017),     # tke
        :X_7 => FT(-56.88902282714844),     # z_obu
        :X_8 => FT(-7.018703460693359)      # res_obu
    )
    
    stds = Dict{Symbol, FT}(
        :X_1 => FT(42.38632678985596),        # mix_len_pi1
        :X_2 => FT(1158.9794158935547),       # mix_len_pi2
        :X_3 => FT(55.166049003601074),        # mix_len_pi3
        :X_4 => FT(0.0010860399197554216),     # bgrad
        :X_5 => FT(0.0001870773485279642),     # strain
        :X_6 => FT(2.441168576478958),          # tke
        :X_7 => FT(4223.447265625),            # z_obu
        :X_8 => FT(414.67926025390625)         # res_obu
    )

    # Normalizing the variables
    X_1 = (X_1 - means[:X_1]) / stds[:X_1]
    X_2 = (X_2 - means[:X_2]) / stds[:X_2]
    X_3 = (X_3 - means[:X_3]) / stds[:X_3]
    X_4 = (X_4 - means[:X_4]) / stds[:X_4]
    X_5 = (X_5 - means[:X_5]) / stds[:X_5]
    X_6 = (X_6 - means[:X_6]) / stds[:X_6]
    X_7 = (X_7 - means[:X_7]) / stds[:X_7]
    X_8 = (X_8 - means[:X_8]) / stds[:X_8]

    X = [X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8]


    # arc = [length(X), 15, 10, 5, 1]
    arc = [length(X), 20, 15, 10, 1]

    # nn_model = construct_fully_connected_nn(arc, param_vec; biases_bool = true, output_layer_activation_function = Flux.identity)
    nn_model = construct_fully_connected_nn(arc, param_vec; biases_bool = true, activation_function = Flux.leakyrelu, output_layer_activation_function = Flux.identity)

    # l_limited = max(nn_model([X]), 0.0)
    l_limited_norm = nn_model(X)[1]
    l_limited = max(FT(l_limited_norm) * FT(510.1035690307617) + FT(36.83180618286133), FT(0.0))

    


    N_eff = sqrt(max(ᶜlinear_buoygrad, 0))

    l_smag = smagorinsky_lilly_length(
        CAP.c_smag(params),
        N_eff,
        ᶜdz,
        ᶜPr,
        ᶜstrain_rate_norm,
    )
    l_limited = max(l_smag, min(l_limited, l_z))

    if ᶜtke < FT(0.001)
        l_limited = FT(0.0)
    end

    l_W = 0.0
    l_TKE = 0.0
    l_N = 0.0

    return MixingLength{FT}(l_limited, l_W, l_TKE, l_N)
end


"""
    mixing_length(params, ustar, ᶜz, sfc_tke, ᶜlinear_buoygrad, ᶜtke, obukhov_length, ᶜstrain_rate_norm, ᶜPr, ᶜtke_exch)

where:
- `params`: set with model parameters
- `ustar`: friction velocity
- `ᶜz`: height
- `tke_sfc`: env kinetic energy at first cell center
- `ᶜlinear_buoygrad`: buoyancy gradient
- `ᶜtke`: env turbulent kinetic energy
- `obukhov_length`: surface Monin Obukhov length
- `ᶜstrain_rate_norm`: Frobenius norm of strain rate tensor
- `ᶜPr`: Prandtl number
- `ᶜtke_exch`: subdomain exchange term

Returns mixing length as a smooth minimum between
wall-constrained length scale,
production-dissipation balanced length scale,
effective static stability length scale, and
Smagorinsky length scale.
"""
function mixing_length(
    params,
    ustar::FT,
    ᶜz::FT,
    z_sfc::FT,
    ᶜdz::FT,
    sfc_tke::FT,
    ᶜlinear_buoygrad::FT,
    ᶜtke::FT,
    obukhov_length::FT,
    ᶜstrain_rate_norm::FT,
    ᶜPr::FT,
    ᶜtke_exch::FT,
    ᶜwʲ::FT,
    ᶜw⁰::FT,
    ::PhysicalMixingLengthModel,
) where {FT}

    turbconv_params = CAP.turbconv_params(params)
    c_m = CAP.tke_ed_coeff(turbconv_params)
    c_d = CAP.tke_diss_coeff(turbconv_params)
    smin_ub = CAP.smin_ub(turbconv_params)
    smin_rm = CAP.smin_rm(turbconv_params)
    c_b = CAP.static_stab_coeff(turbconv_params)
    vkc = CAP.von_karman_const(params)

    param_vec = CAP.mixing_length_param_vec(params)

    # compute the maximum mixing length at height z
    l_z = ᶜz - z_sfc

    # compute the l_W - the wall constraint mixing length
    # which imposes an upper limit on the size of eddies near the surface
    # kz scale (surface layer)
    if obukhov_length < 0.0 #unstable
        l_W =
            vkc * (ᶜz - z_sfc) /
            max(sqrt(sfc_tke / ustar / ustar) * c_m, eps(FT)) *
            min((1 - 100 * (ᶜz - z_sfc) / obukhov_length)^FT(0.2), 1 / vkc)
    else # neutral or stable
        l_W =
            vkc * (ᶜz - z_sfc) /
            max(sqrt(sfc_tke / ustar / ustar) * c_m, eps(FT))
    end

    # compute l_TKE - the production-dissipation balanced length scale
    a_pd = c_m * (2 * ᶜstrain_rate_norm - ᶜlinear_buoygrad / ᶜPr) * sqrt(ᶜtke)
    # Dissipation term
    c_neg = c_d * ᶜtke * sqrt(ᶜtke)
    if abs(a_pd) > eps(FT) && 4 * a_pd * c_neg > -(ᶜtke_exch * ᶜtke_exch)
        l_TKE = max(
            -(ᶜtke_exch / 2 / a_pd) +
            sqrt(ᶜtke_exch * ᶜtke_exch + 4 * a_pd * c_neg) / 2 / a_pd,
            0,
        )
    elseif abs(a_pd) < eps(FT) && abs(ᶜtke_exch) > eps(FT)
        l_TKE = c_neg / ᶜtke_exch
    else
        l_TKE = FT(0)
    end

    # compute l_N - the effective static stability length scale.
    N_eff = sqrt(max(ᶜlinear_buoygrad, 0))
    if N_eff > 0.0
        l_N = min(sqrt(max(c_b * ᶜtke, 0)) / N_eff, l_z)
    else
        l_N = l_z
    end

    # compute l_smag - smagorinsky length scale
    l_smag = smagorinsky_lilly_length(
        CAP.c_smag(params),
        N_eff,
        ᶜdz,
        ᶜPr,
        ᶜstrain_rate_norm,
    )

    # add limiters
    l = SA.SVector(
        l_N > l_z ? l_z : l_N,
        l_TKE > l_z ? l_z : l_TKE,
        l_W > l_z ? l_z : l_W,
    )
    # get soft minimum
    l_smin = lamb_smooth_minimum(l, smin_ub, smin_rm)
    l_limited = max(l_smag, min(l_smin, l_z))

    # ln_l_bias = param_vec[1].* ((ᶜlinear_buoygrad .- 0.00026)/0.0003)
    #     .+ param_vec[2].*abs((ᶜtke .- 0.11454)/0.34)
    #     .+ param_vec[3]*((ᶜstrain_rate_norm .- 1.5e-5)/8.03e-5)

    # l_limited = l_limited + exp(ln_l_bias)

    return MixingLength{FT}(l_limited, l_W, l_TKE, l_N)
end

"""
    turbulent_prandtl_number(params, obukhov_length, ᶜRi_grad)

where:
- `params`: set with model parameters
- `obukhov_length`: surface Monin Obukhov length
- `ᶜRi_grad`: gradient Richardson number

Returns the turbulent Prandtl number give the obukhov length sign and
the gradient Richardson number, which is calculated from the linearized
buoyancy gradient and shear production.
"""
function turbulent_prandtl_number(
    params,
    obukhov_length,
    ᶜlinear_buoygrad,
    ᶜstrain_rate_norm,
)
    FT = eltype(params)
    turbconv_params = CAP.turbconv_params(params)
    Ri_c = CAP.Ri_crit(turbconv_params)
    ω_pr = CAP.Prandtl_number_scale(turbconv_params)
    Pr_n = CAP.Prandtl_number_0(turbconv_params)
    ᶜRi_grad = min(ᶜlinear_buoygrad / max(2 * ᶜstrain_rate_norm, eps(FT)), Ri_c)
    if obukhov_length > 0 && ᶜRi_grad > 0 #stable
        # CSB (Dan Li, 2019, eq. 75), where ω_pr = ω_1 + 1 = 53.0 / 13.0
        prandtl_nvec =
            Pr_n * (
                2 * ᶜRi_grad / (
                    1 + ω_pr * ᶜRi_grad -
                    sqrt((1 + ω_pr * ᶜRi_grad)^2 - 4 * ᶜRi_grad)
                )
            )
    else
        prandtl_nvec = Pr_n
    end
    return prandtl_nvec
end

edmfx_filter_tendency!(Yₜ, Y, p, t, turbconv_model) = nothing

"""
   Apply EDMF filters:
   - Relax u_3 to zero when it is negative
   - Relax ρa to zero when it is negative
"""
function edmfx_filter_tendency!(Yₜ, Y, p, t, turbconv_model::PrognosticEDMFX)

    n = n_mass_flux_subdomains(turbconv_model)
    (; dt) = p

    if p.atmos.edmfx_model.filter isa Val{true}
        for j in 1:n
            @. Yₜ.f.sgsʲs.:($$j).u₃ -=
                C3(min(Y.f.sgsʲs.:($$j).u₃.components.data.:1, 0)) / float(dt)
            @. Yₜ.c.sgsʲs.:($$j).ρa -= min(Y.c.sgsʲs.:($$j).ρa, 0) / float(dt)
        end
    end
end

"""
    Count number of parameters in fully-connected NN model given Array specifying architecture following
        the pattern: [#inputs, #neurons in L1, #neurons in L2, ...., #outputs]. Equal to the number of weights + biases.
"""
num_params_from_arc(nn_arc::AbstractArray{Int}) = num_weights_from_arc(nn_arc) + num_biases_from_arc(nn_arc)

"""
    Count number of weights in fully-connected NN architecture.
"""
num_weights_from_arc(nn_arc::AbstractArray{Int}) = sum(i -> nn_arc[i] * nn_arc[i + 1], 1:(length(nn_arc) - 1))

"""
    Count number of biases in fully-connected NN architecture.
"""
num_biases_from_arc(nn_arc::AbstractArray{Int}) = sum(i -> nn_arc[i + 1], 1:(length(nn_arc) - 1))


"""
    construct_fully_connected_nn(
        arc::AbstractArray{Int},
        params::AbstractArray{FT};
        biases_bool::bool = false,
        activation_function::Flux.Function = Flux.relu,
        output_layer_activation_function::Flux.Function = Flux.relu,)

    Given network architecture and parameter vectors, construct NN model and unpack weights (and biases if `biases_bool` is true).
    - `arc` :: vector specifying network architecture
    - `params` :: parameter vector containing weights (and biases if `biases_bool` is true)
    - `biases_bool` :: bool specifying whether `params` includes biases.
    - `activation_function` :: activation function for hidden layers
    - `output_layer_activation_function` :: activation function for output layer
"""
function construct_fully_connected_nn(
    arc::AbstractArray{Int},
    params::AbstractArray{FT};
    biases_bool::Bool = false,
    activation_function::Flux.Function = Flux.relu,
    output_layer_activation_function::Flux.Function = Flux.relu,
) where {FT <: Real}

    # check consistency of architecture and parameters
    if biases_bool
        n_params_nn = num_params_from_arc(arc)
        n_params_vect = length(params)
    else
        n_params_nn = num_weights_from_arc(arc)
        n_params_vect = length(params)
    end
    if n_params_nn != n_params_vect
        error("Incorrect number of parameters ($n_params_vect) for requested NN architecture ($n_params_nn)!")
    end

    layers = []
    parameters_i = 1
    # unpack parameters in parameter vector into network
    for layer_i in 1:(length(arc) - 1)
        if layer_i == length(arc) - 1
            activation_function = output_layer_activation_function
        end
        layer_num_weights = arc[layer_i] * arc[layer_i + 1]

        nn_biases = if biases_bool
            params[(parameters_i + layer_num_weights):(parameters_i + layer_num_weights + arc[layer_i + 1] - 1)]
        else
            biases_bool
        end

        layer = Flux.Dense(
            reshape(params[parameters_i:(parameters_i + layer_num_weights - 1)], arc[layer_i + 1], arc[layer_i]),
            nn_biases,
            activation_function,
        )
        parameters_i += layer_num_weights

        if biases_bool
            parameters_i += arc[layer_i + 1]
        end
        push!(layers, layer)
    end

    return Flux.Chain(layers...)
end

"""
serialize_ml_model(
    ml_model::Flux.Chain,
    )

    Given Flux NN model, serialize model weights into a vector of parameters.
    - `ml_model` - A Flux model instance.
"""

function serialize_ml_model(ml_model::Flux.Chain)
    parameters = []
    param_type = eltype.(Flux.params(ml_model))[1]
    for layer in ml_model
        for param in Flux.params(layer)
            param_flattened = reshape(param, length(param))
            push!(parameters, param_flattened...)
        end
    end
    return convert(Array{param_type}, parameters)
end


"""
    construct_fully_connected_nn_default(
        arc::AbstractArray{Int};
        biases_bool::Bool = false,
        activation_function::Flux.Function = Flux.relu,
        output_layer_activation_function::Flux.Function = Flux.relu,
    )

Given network architecture, construct NN model with default `Flux.jl` weight initialization (glorot_uniform).
- `arc` :: vector specifying network architecture
- `biases_bool` :: bool specifying whether to include biases.
- `activation_function` :: activation function for hidden layers
- `output_layer_activation_function` :: activation function for output layer
"""

function construct_fully_connected_nn_default(
    arc::AbstractArray{Int};
    biases_bool::Bool = false,
    activation_function::Flux.Function = Flux.relu,
    output_layer_activation_function::Flux.Function = Flux.relu,
)

    layers = []
    for layer_i in 1:(length(arc) - 1)
        if layer_i == length(arc) - 1
            activation_function = output_layer_activation_function
        end

        layer = Flux.Dense(arc[layer_i] => arc[layer_i + 1], activation_function; bias = biases_bool)
        push!(layers, layer)
    end

    return Flux.Chain(layers...)
end
