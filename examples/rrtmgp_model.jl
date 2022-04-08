push!(LOAD_PATH, joinpath(@__DIR__, "..", ".."))

using RRTMGP
using CLIMAParameters
using ClimaCore.Spaces: FiniteDifferenceSpace

using Pkg
using NCDatasets
using UnPack

# zᵢ is the elevation of layer i
#
# Reversible adiabatic process equation + definition of adiabatic index:
# P(z)^{1-γ}T(z)^γ = constant ==>
#     T(z)^{γ/(γ-1)}/P(z) = constant ==>
#     T(0)^{γ/(γ-1)}/P(0) = T(z₁)^{γ/(γ-1)}/P(z₁) ==>
#     P(0) = P(z₁)(T(0)/T(z₁))^{γ/(γ-1)}
# cᵥ = R/(γ-1), γ = cₚ/cᵥ ==>
#     γ/(γ-1) = cₚ/R ==>
#   * P(0) = P(z₁)(T(0)/T(z₁))^{cₚ/R}
#
# Hydrostatic balance equation + dry ideal gas law + temperature interpolation:
# ∂P(z)/∂z = -gρ(z)
# ρ(z) = P(z)/RT(z) ==>
#     ∂P(z)/∂z = -gP(z)/RT(z) ==>
#     1/P(z) ∂P(z)/∂z = -g/RT(z) ==>
#     ∫_0^{z₁} 1/P(z) ∂P(z)/∂z dz = -g/R ∫_0^{z₁} 1/T(z) dz ==>
#     ∫_{P(0)}^{P(z₁)} 1/u du = -g/R ∫_0^{z₁} 1/T(z) dz ==>
#     log(P(z₁)/P(0)) = -g/R ∫_0^{z₁} 1/T(z) dz ==>
#     P(0) = P(z₁) exp(g/R ∫_0^{z₁} 1/T(z) dz)
# T(z) = T(0) - (T(0) - T(z₁))z/z₁ ==>
#     ∫_0^{z₁} 1/T(z) dz = z₁/(T(0) - T(z₁)) log(T(0)/T(z₁)) ==>
#     P(0) = P(z₁) exp(gz₁/(R(T(0) - T(z₁))) log(T(0)/T(z₁))) ==>
#   * P(0) = P(z₁)(T(0)/T(z₁))^(gz₁/(R(T(0) - T(z₁))))
#
# Combining equations for P(0):
# P(z₁)(T(0)/T(z₁))^{cₚ/R} = P(z₁)(T(0)/T(z₁))^(gz₁/(R(T(0) - T(z₁)))) ==>
#     cₚ/R = gz₁/(R(T(0) - T(z₁))) ==>
#   * T(0) = T(z₁) + gz₁/cₚ ==>
#   * P(0) = P(z₁)(1 + gz₁/(cₚT(z₁)))^{cₚ/R}


#  |     ----     |     ----     |
# lev1   lay1    lev2   lay2    lev3
#
# :arithmetic_mean_pressure
#     Interpolation:
#         plev2 = (play1 + play2)/2
#         tlev2 = (tlay1 + tlay2)/2
#     Extrapolation:
#         "play3" = play2 + (play2 - play1) = 2 play2 - play1 ==>
#             plev3 = (play2 + "play3")/2 = (3 play2 - play1)/2
#         play1 = 2 plev2 - play2 ==>
#             plev3 = (4 play2 - 2 plev2)/2 = 2 play2 - plev2
#         "tlay3" = tlay2 + (tlay2 - tlay1) = 2 tlay2 - tlay1 ==>
#             tlev3 = (tlay2 + "tlay3")/2 = (3 tlay2 - tlay1)/2
#         tlay1 = 2 tlev2 - tlay2 ==>
#             tlev3 = (4 tlay2 - 2 tlev2)/2 = 2 tlay2 - tlev2
# :geometric_mean_pressure
#     Interpolation:
#         plev2 = sqrt(play1 play2)
#         tlev2 = (tlay1 + tlay2)/2
#     Extrapolation:
#         "play3" = play2 (play2/play1) = play2^2/play1 ==>
#             plev3 = sqrt(play2 "play3") = sqrt(play2^3/play1)
#         play1 = plev2^2/play2 ==>
#             plev3 = sqrt(play2^4/plev2^2) = play2^2/plev2
#         "tlay3" = tlay2 + (tlay2 - tlay1) = 2 tlay2 - tlay1 ==>
#             tlev3 = (tlay2 + "tlay3")/2 = (3 tlay2 - tlay1)/2
#         tlay1 = 2 tlev2 - tlay2 ==>
#             tlev3 = (4 tlay2 - 2 tlev2)/2 = 2 tlay2 - tlev2
# :pressure_interpolation
#     Interpolation:
#         t(z) = tlay1 + L (z - zlay1)
#         t(zlay2) = tlay2 ==>
#             tlay1 + L (zlay2 - zlay1) = tlay2 ==>
#             L = (tlay2 - tlay1)/(zlay2 - zlay1)
#         tlev2 = t(zlev2) =
#               = tlay1 + (tlay2 - tlay1)(zlev2 - zlay1)/(zlay2 - zlay1)
#         If tlay2 ≠ tlay1:
#             p(z) = play1 (t(z)/tlay1)^C
#             p(zlay2) = play2 ==>
#                 play1 (tlay2/tlay1)^C = play2 ==>
#                 (tlay2/tlay1)^C = play2/play1 ==>
#                 C log(tlay2/tlay1) = log(play2/play1) ==>
#                 C = log(play2/play1)/log(tlay2/tlay1)
#             plev2 = p(zlev2) =
#                   = play1 (tlev2/tlay1)^(log(play2/play1)/log(tlay2/tlay1)) =
#                   = play1 (play2/play1)^(log(tlev2/tlay1)/log(tlay2/tlay1))
#         If tlay2 = tlay1:
#             p(z) = play1 exp(D (z - zlay1))
#             p(zlay2) = play2 ==>
#                 play1 exp(D (zlay2 - zlay1)) = play2 ==>
#                 exp(D (zlay2 - zlay1)) = play2/play1 ==>
#                 D = log(play2/play1)/(zlay2 - zlay1)
#             plev2 = p(zlev2) =
#                   = play1 exp(log(play2/play1)(zlev2 - zlay1)/(zlay2 - zlay1)) =
#                   = play1 (play2/play1)^((zlev2 - zlay1)/(zlay2 - zlay1))
#     Extrapolation:
#         tlev3 = t(zlev3) =
#               = tlay1 + (tlay2 - tlay1)(zlev3 - zlay1)/(zlay2 - zlay1)
#         If tlay2 ≠ tlay1:
#             plev3 = p(zlev3) =
#                   = play1 (play2/play1)^(log(tlev3/tlay1)/log(tlay2/tlay1))
#         If tlay2 = tlay1:
#             plev3 = p(zlev3) =
#                   = play1 (play2/play1)^((zlev3 - zlay1)/(zlay2 - zlay1))

#
# Alternatives for determining T(0)
#     • Distance Extrapolation:
#           T(z) = T(z₁) - (T(z₁) - T(z₂))(z - z₁)/(z₂ - z₁) ==>
#               T(0) = T(z₁) + (T(z₁) - T(z₂))z₁/(z₂ - z₁)
#           z₂ = 3z₁ (equally spaced cells) ==>
#               T(0) = (3T(z₁) - T(z₂))/2
#     • Surface Temperature Assumption:
#           T(0) = T_{sfc}
#     • Adiabatic Process and Hydrostatic Balance Assumptions:
#           T(0) = T(z₁) + gz₁/cₚ
# We could also use pressure extrapolation to determine T(0), which would
# require combining one of the two equations for P(0) given above with
# T(0) = T(z₁) - (T(z₁) - T(z₂))(P(0) - P(z₁))/(P(z₂) - P(z₁)).
#
# Alternatives for determining P(0)
#     • Distance Extrapolation:
#           P(z) = P(z₁) - (P(z₁) - P(z₂))(z - z₁)/(z₂ - z₁) ==>
#               P(0) = P(z₁) + (P(z₁) - P(z₂))z₁/(z₂ - z₁)
#           z₂ = 3z₁ (equally spaced cells) ==>
#               P(0) = (3P(z₁) - P(z₂))/2
#     • Adiabatic Process and Some Assumption About T(0) (Extrap. or T_{sfc}):
#           P(0) = P(z₁)(T(0))/T(z₁))^{cₚ/R}
#     • Hydrostatic Balance and Some Assumption About T(0) (Extrap. or T_{sfc}):
#           P(0) = P(z₁)(T(0)/T(z₁))^(gz₁/(R(T(0) - T(z₁))))
#     • Adiabatic Process and Hydrostatic Balance Assumptions:
#           P(0) = P(z₁)(1 + gz₁/(cₚT(z₁)))^{cₚ/R}
#
# Determining T(zₜₒₚ) with distance extrapolation:
#     T(z) = T(zₙ₋₁) - (T(zₙ₋₁) - T(zₙ))(z - zₙ₋₁)/(zₙ - zₙ₋₁) ==>
#         T(zₜₒₚ) = T(zₙ₋₁) - (T(zₙ₋₁) - T(zₙ))(zₜₒₚ - zₙ₋₁)/(zₙ - zₙ₋₁)
#     zₜₒₚ - zₙ₋₁ = 3(zₙ - zₙ₋₁)/2 (equally spaced cells) ==>
#         T(zₜₒₚ) = (3T(zₙ) - T(zₙ₋₁))/2

# Stores the RRTMGP.RTESolver.Solver, along with everything needed to use it.
# Provides a simple, informative interface that requires no knowledge of the
# solver's internal structure.
struct RRTMGPModel{
    Optics,
    No_LW,
    No_SW,
    Update_BL,
    ST,
    LT,
    PT,
    OT,
    MT,
}
    solver::ST
    lookups::LT
    param_set::PT
    max_threads::Int
    level_olators::OT
    mutable_values::MT
end

function Base.getproperty(model::RRTMGPModel, s::Symbol)
    if s in fieldnames(typeof(model))
        return getfield(model, s)
    else
        return getproperty(getfield(model, :mutable_values), s)
    end
end

function Base.propertynames(model::RRTMGPModel, private::Bool = false)
    names = propertynames(getfield(model, :mutable_values))
    if private
        return (fieldnames(typeof(model)), names...)
    else
        return names
    end
end

abstract type AbstractCoordsProvided{Pass} end
struct NoCoordsProvided{Pass} <: AbstractCoordsProvided{Pass} end
struct VertCoordsProvided{Pass, T} <: AbstractCoordsProvided{Pass}
    zs::T
end
# TODO: Implement this once 3D spaces are available.
struct AllCoordsProvided{Pass} <: AbstractCoordsProvided{Pass} end

function coordinate_info(pass, DA, add_boundary, extension, args...)
    FT, ncol, nlay_domain, domain_coords = domain_info(pass, DA, args...)
    nlay_extension, extension_coords =
        extension_info(pass, DA, FT, ncol, extension)
    nlay_boundary = add_boundary ? 1 : 0
    return (FT, ncol, nlay_domain + nlay_extension + nlay_boundary, (;
        domain = domain_coords,
        extension = extension_coords,
        nlay_boundary = nlay_boundary,
        nlay_ghost = nlay_extension + nlay_boundary,
    ))
end

domain_info(pass, DA, FT::Type{<:AbstractFloat}, nlay::Int, ncol::Int = 1) =
    (FT, ncol, nlay, NoCoordsProvided{pass}())
function domain_info(pass, DA, z_lev::AbstractArray{<:AbstractFloat, 2})
    zs = (;
        lay = DA((z_lev[1:end - 1, :] .+ z_lev[2:end, :]) ./ 2),
        lev = DA(z_lev),
    )
    return (
        eltype(z_lev),
        size(z_lev, 2),
        size(z_lev, 1) - 1,
        VertCoordsProvided{pass, typeof(zs)}(zs),
    )
end
domain_info(pass, DA, z_lev::AbstractArray{<:AbstractFloat, 1}, ncol::Int = 1) =
    domain_info(pass, DA, repeat(z_lev, 1, ncol))
# TODO: Use coordinates_data
domain_info(pass, DA, space::FiniteDifferenceSpace, ncol::Int = 1) =
    domain_info(pass, DA, collect(space.mesh.faces), ncol)

extension_info(pass, DA, FT, ncol, nlay::Int) = (nlay, NoCoordsProvided{pass}())
function extension_info(
    pass,
    DA,
    FT,
    ncol,
    z_lev::AbstractArray{<:AbstractFloat, 2},
)
    if eltype(z_lev) != FT
        throw(ArgumentError(string(
            "domain coordinates and extension coordinates must have the same ",
            "types, but they have types $FT and $(eltype(z_lev)), respectively",
        )))
    end
    if size(z_lev, 2) != ncol
        throw(ArgumentError(string(
            "domain coordinates and extension coordinates must have the same ",
            "number of columns, but they have $ncol and $(size(z_lev, 2)) ",
            "columns, respectively",
        )))
    end
    zs = (;
        lay = DA((z_lev[1:end - 1, :] .+ z_lev[2:end, :]) ./ 2),
        lev = DA(z_lev),
    )
    return (size(z_lev, 1) - 1, VertCoordsProvided{pass, typeof(zs)}(zs))
end
extension_info(pass, DA, FT, ncol, z_lev::AbstractArray{<:AbstractFloat, 1}) =
    extension_info(pass, DA, FT, ncol, repeat(z_lev, 1, ncol))
extension_info(pass, DA, FT, ncol, space::FiniteDifferenceSpace) =
    extension_info(pass, DA, FT, ncol, collect(space.mesh.faces))

function rrtmgp_artifact(subfolder, file_name)
    artifact_name = "RRTMGPReferenceData"
    artifacts_file =
        joinpath(dirname(dirname(pathof(RRTMGP))), "Artifacts.toml")
    data_folder = joinpath(
        Pkg.Artifacts.ensure_artifact_installed(artifact_name, artifacts_file),
        artifact_name,
    )
    return Dataset(joinpath(data_folder, subfolder, file_name), "r")
end

function set_and_record_array!(
    array,
    name,
    vertical_symbol,
    rec,
    coords,
    init_dict,
)
    symbol = Symbol(name)
    if !(symbol in keys(init_dict))
        throw(UndefKeywordError(symbol))
    end
    value = pop!(init_dict, symbol)
    if !(value isa Union{Real, AbstractArray{<:Real}, Function})
        throw(ArgumentError(string(
            "keyword argument $symbol must be a Real, an ",
            "AbstractArray{<:Real}, or a Function",
        )))
    end

    ext_symbol = Symbol("extension_" * name)
    if ext_symbol in keys(init_dict)
        if vertical_symbol == :NA
            throw(ArgumentError(
                "got unsupported keyword argument $ext_symbol",
            ))
        end
        if coords.nlay_ghost == coords.nlay_boundary
            throw(ArgumentError(string(
                "keyword argument $ext_symbol cannot be used if an extension ",
                "is not specified",
            )))
        end
        ext_value = pop!(init_dict, ext_symbol)
        if !(ext_value isa Union{Real, AbstractArray{<:Real}, Function})
            throw(ArgumentError(string(
                "keyword argument $ext_symbol must be a Real, an ",
                "AbstractArray{<:Real}, or a Function",
            )))
        end
    else
        ext_value = value
    end

    set_arrays!(array, value, ext_value, vertical_symbol, coords)
    record_array!(rec, array, name, vertical_symbol, coords)
end

function record_array!(rec, array, name, vertical_symbol, coords)
    if vertical_symbol == :NA || coords.nlay_ghost == 0
        push!(rec, (Symbol(name), array))
    else
        push!(rec, (Symbol(name), @view(array[1:end - coords.nlay_ghost, :])))
    end
end

function set_arrays!(array, value, ext_value, vertical_symbol, coords)
    if vertical_symbol == :NA
        set_array!(array, value, coords.domain)
    elseif coords.nlay_ghost == 0
        set_array!(array, value, coords.domain, vertical_symbol)
    else
        domain_array = @view array[1:end - coords.nlay_ghost, :]
        set_array!(domain_array, value, coords.domain, vertical_symbol)
        if coords.nlay_ghost != coords.nlay_boundary
            ext_array = @view(
                array[end - coords.nlay_ghost + 1:end - coords.nlay_boundary, :]
            )
            set_array!(ext_array, ext_value, coords.extension, vertical_symbol)
        end
    end
end

# The following 3 functions are identical to
#     set_array!(array, value::Union{Real, AbstractArray{<:Real}}, args...) =
#         array .= value
# but they also allow `array` to be a CuArray while `value` is an Array (in
# which case broadcasting results in an error).
# TODO: Can this be parallelized?
set_array!(array, value::Real, args...) = fill!(array, value)
function set_array!(array, value::AbstractArray{<:Real, 1}, args...)
    if size(value, 1) == size(array, 1)
        for col in eachcol(array)
            copyto!(col, value)
        end
    else
        throw(ArgumentError(string(
            "expected array of size ($(size(array, 1)),); ",
            "received array of size $(size(value))",
        )))
    end
end
function set_array!(array, value::AbstractArray{<:Real, 2}, args...)
    if size(value) == size(array)
        copyto!(array, value)
    elseif size(value) == (1, size(array, 2))
        for (icol, col) in enumerate(eachcol(array))
            fill!(col, value[1, icol])
        end
    else
        throw(ArgumentError(string(
            "expected array of size $(size(array)); ",
            "received array of size $(size(value))",
        )))
    end
end
set_array!(array, value::Function, args...) =
    set_array_fun!(array, value, args...)

#                              | (ncol,) | (ngpt/nbnd, ncol) | (nlev/nlay, ncol)
# -----------------------------|---------|-------------------|------------------
# NoCoordsProvided   | No Pass | f()     | f(igpt)           | f(ilev)
#                    | Pass    | f(icol) | f(igpt, icol)     | f(ilev, icol)
# VertCoordsProvided | No Pass | f()     | f(igpt)           | f(z)
#                    | Pass    | f(icol) | f(igpt, icol)     | f(z, icol)
# AllCoordsProvided  | No Pass | f()     | f(igpt)           | f(z)
#                    | Pass    | f(x, y) | f(igpt, x, y)     | f(x, y, z)
# Maybe replace f(x, y) and f(igpt, x, y) with f(lat, lon) and f(igpt, lat, lon)
# when implementing them?
set_array_fun!(array, f, coords::AbstractCoordsProvided{false}) =
    ndims(array) == 1 ? fill!(array, f()) : array .= f.(1:size(array, 1))
set_array_fun!(array, f, coords::AbstractCoordsProvided{true}) =
    ndims(array) == 1 ? array .= f.(1:size(array, 1)) :
        array .= f.(1:size(array, 1), (1:size(array, 2))')
set_array_fun!(array, f, coords::NoCoordsProvided{false}, vertical_symbol) =
    array .= f.(1:size(array, 1))
set_array_fun!(array, f, coords::NoCoordsProvided{true}, vertical_symbol) =
    array .= f.(1:size(array, 1), (1:size(array, 2))')
set_array_fun!(array, f, coords::VertCoordsProvided{false}, vertical_symbol) =
    array .= f.(getproperty(coords.zs, vertical_symbol))
set_array_fun!(array, f, coords::VertCoordsProvided{true}, vertical_symbol) =
    array .= f.(getproperty(coords.zs, vertical_symbol), (1:size(array, 2))')
set_array_fun!(array, f, coords::AllCoordsProvided, args...) =
    @error "`set_array_fun!` is not implemented for type AllCoordsProvided"

struct LevelInterpolator{Mode, VT, VTN, FTN, ATN}
    p::VT
    pꜜ::VT
    pꜛ::VTN
    t::VT
    tꜜ::VT
    tꜛ::VTN
    precomputed_constant::FTN
    precomputed_z_values::ATN
end

struct LevelExtrapolator{Mode, VT, VTN, FTN, ATN}
    p::VT
    p⁺::VT
    p⁺⁺::VTN
    t::VT
    t⁺::VT
    t⁺⁺::VTN
    precomputed_constant::FTN
    precomputed_z_values::ATN
end

function level_olator(
    is_interp,
    lev_range,
    lay_range1,
    lay_range2,
    mode,
    p_lay,
    p_lev,
    t_lay,
    t_lev,
    # coords,
    z_lay,
    z_lev,
    param_set,
)
    if length(lev_range) == 0
        lev_range = lev_range[1]
        lay_range1 = lay_range1[1]
        lay_range2 = lay_range2[1]
    end

    p = @view p_lev[lev_range, :]
    t = @view t_lev[lev_range, :]
    p1 = @view p_lay[lay_range1, :]
    t1 = @view t_lay[lay_range1, :]

    if mode == :ideal_coefs
        p2 = nothing
        t2 = nothing
        g = CLIMAParameters.Planet.grav(param_set)
        cₚ = CLIMAParameters.Planet.cp_d(param_set)
        R = CLIMAParameters.Planet.R_d(param_set)
        precomputed_constant = cₚ / R
        # if coords.domain isa NoCoordsProvided
        #     throw(ArgumentError(
        #         "domain coordinates must be provided to use :ideal_coefs mode",
        #     ))
        # end
        # z = @view coords.domain.zs.lev[lev_range, :]
        # z1 = @view coords.domain.zs.lay[lay_range1, :]
        z = @view z_lev[lev_range, :]
        z1 = @view z_lay[lay_range1, :]
        precomputed_z_values = g / cₚ .* (z1 .- z)
    else
        p2 = @view p_lay[lay_range2, :]
        t2 = @view t_lay[lay_range2, :]
        precomputed_constant = nothing
        if mode == :best_fit
            # if coords.domain isa NoCoordsProvided
            #     throw(ArgumentError(
            #         "domain coordinates must be provided to use :best_fit mode",
            #     ))
            # end
            # z = @view coords.domain.zs.lev[lev_range, :]
            # z1 = @view coords.domain.zs.lay[lay_range1, :]
            # z2 = @view coords.domain.zs.lay[lay_range2, :]
            z = @view z_lev[lev_range, :]
            z1 = @view z_lay[lay_range1, :]
            z2 = @view z_lay[lay_range2, :]
            precomputed_z_values = (z .- z1) ./ (z2 .- z1)
        else
            precomputed_z_values = nothing
        end
    end
    
    return (is_interp ? LevelInterpolator : LevelExtrapolator){
        mode,
        typeof(p),
        typeof(p2),
        typeof(precomputed_constant),
        typeof(precomputed_z_values),
    }(p, p1, p2, t, t1, t2, precomputed_constant, precomputed_z_values)
end

# NOTE: :geometric_mean, :uniform_z, and :uniform_p are all special cases of
#       :best_fit (they assume different values for z, rather than using the
#       true values), but :average and :ideal_coefs are not consistent with
#       :best_fit.
# NOTE: p⁺⁺ and t⁺⁺ could be switched from layers to levels, which would make
#       the extrapolation code a little simpler, but that would make it harder
#       to understand what's going on.
# TODO: Should uniform_z_p and best_fit_p be kernel functions?

uniform_z_p(t, t1, t2, p1, p2) = t1 == t2 ?
    sqrt(p1 * p2) : # p1 * (p2 / p1)^0.5
    p1 * (p2 / p1)^(log(t / t1) / log(t2 / t1))
best_fit_p(t, t1, t2, p1, p2, precomputed_z_value) = t1 == t2 ?
    p1 * (p2 / p1)^precomputed_z_value :
    p1 * (p2 / p1)^(log(t / t1) / log(t2 / t1))

function (l::LevelInterpolator{:average})()
    l.t .= (l.tꜜ .+ l.tꜛ) ./ 2
    l.p .= (l.pꜜ .+ l.pꜛ) ./ 2
end
function (l::LevelInterpolator{:geometric_mean})()
    l.t .= sqrt.(l.tꜜ .* l.tꜛ)
    l.p .= sqrt.(l.pꜜ .* l.pꜛ)
end
function (l::LevelInterpolator{:uniform_z})()
    l.t .= (l.tꜜ .+ l.tꜛ) ./ 2
    l.p .= uniform_z_p.(l.t, l.tꜜ, l.tꜛ, l.pꜜ, l.pꜛ)
end
function (l::LevelInterpolator{:uniform_p})()
    l.p .= (l.pꜜ .+ l.pꜛ) ./ 2
    l.t .= l.tꜜ .* (l.tꜛ ./ l.tꜜ).^(log.(l.p ./ l.pꜜ) ./ log.(l.pꜛ ./ l.pꜜ))
end
function (l::LevelInterpolator{:best_fit})()
    l.t .= l.tꜜ .+ (l.tꜛ .- l.tꜜ) .* l.precomputed_z_values
    l.p .= best_fit_p.(l.t, l.tꜜ, l.tꜛ, l.pꜜ, l.pꜛ, l.precomputed_z_values)
end

function (l::LevelExtrapolator{:average})()
    l.t .= (3 .* l.t⁺ .- l.t⁺⁺) ./ 2
    l.p .= (3 .* l.p⁺ .- l.p⁺⁺) ./ 2
end
function (l::LevelExtrapolator{:geometric_mean})()
    l.t .= sqrt.(l.t⁺.^3 ./ l.t⁺⁺)
    l.p .= sqrt.(l.p⁺.^3 ./ l.p⁺⁺)
end
function (l::LevelExtrapolator{:uniform_z})()
    l.t .= (3 .* l.t⁺ .- l.t⁺⁺) ./ 2
    l.p .= uniform_z_p.(l.t, l.t⁺, l.t⁺⁺, l.p⁺, l.p⁺⁺)
end
function (l::LevelExtrapolator{:uniform_p})()
    l.p .= (3 .* l.p⁺ .- l.p⁺⁺) ./ 2
    l.t .= l.t⁺ .* (l.p ./ l.p⁺).^(log.(l.t⁺⁺ ./ l.t⁺) ./ log.(l.p⁺⁺ ./ l.p⁺))
end
function (l::LevelExtrapolator{:best_fit})()
    l.t .= l.t⁺ .+ (l.t⁺⁺ .- l.t⁺) .* l.precomputed_z_values
    l.p .= best_fit_p.(l.t, l.t⁺, l.t⁺⁺, l.p⁺, l.p⁺⁺, l.precomputed_z_values)
end
# l.t⁺ .+ (l.t⁺⁺ .- l.t⁺) .* (z - z⁺) / (z⁺⁺ - z⁺)
function (l::LevelExtrapolator{:ideal_coefs})()
    l.t .= l.t⁺ .+ l.precomputed_z_values
    l.p .= l.p⁺ .* (l.t ./ l.t⁺).^l.precomputed_constant
end


function RRTMGPModel(
    param_set::CLIMAParameters.AbstractEarthParameterSet,
    args...;
    extension = 0,
    DA::Type{<:AbstractArray} = RRTMGP.Device.array_type(),
    optics::Symbol = :clear,
    level_computation::Symbol = :average,
    use_ideal_coefs_for_bottom_level::Bool = false,
    add_isothermal_boundary_layer::Bool = false,
    disable_longwave::Bool = false,
    disable_shortwave::Bool = false,
    use_one_scalar::Bool = false,
    use_pade_method::Bool = false,
    pass_column_to_function::Bool = true,
    max_threads::Int = 256,
    kwargs...,
)
    optics_modes = (:gray, :clear, :all, :all_with_clear)
    if !(optics in optics_modes)
        throw(ArgumentError(string(
            "keyword argument optics must be set to ",
            join(optics_modes, ", ", ", or "),
        )))
    end
    level_modes =
        (:none, :average, :geometric_mean, :uniform_z, :uniform_p, :best_fit)
    if !(level_computation in level_modes)
        throw(ArgumentError(string(
            "keyword argument level_computation must be set to ",
            join(level_modes, ", ", ", or "),
        )))
    end
    if disable_longwave && disable_shortwave
        throw(ArgumentError(
            "either longwave or shortwave fluxes must be enabled",
        ))
    end
    if use_one_scalar && !disable_shortwave
        @warn string(
            "the OneScalar method for computing fluxes ignores upward ",
            "shortwave fluxes; consider setting `use_one_scalar = false` or ",
            "`disable_shortwave = true`",
        )
    end
    if use_pade_method && !(optics in (:all, :all_with_clear))
        @warn string(
            "the PADE method for computing cloud optical properties is only ",
            "used when `optics = :all` or `optics = :all_with_clear`",
        )
    end

    FT, ncol, nlay, coords = coordinate_info(
        pass_column_to_function,
        DA,
        add_isothermal_boundary_layer,
        extension,
        args...,
    )
    op_symb = use_one_scalar ? :OneScalar : :TwoStream

    init_dict = Dict(kwargs)
    lookups = (;)
    rec = []
    tuple = (rec, coords, init_dict)

    if disable_longwave
        src_lw = flux_lw = fluxb_lw = bcs_lw = nothing
    else
        if optics == :gray
            nbnd_lw = 1
            ngpt_lw = 1
        else
            ds_lw = rrtmgp_artifact("lookup_tables", "clearsky_lw.nc")
            lookup_lw, idx_gases =
                RRTMGP.LookUpTables.LookUpLW(ds_lw, Int, FT, DA)
            close(ds_lw)
            lookups = (; lookups..., lookup_lw, idx_gases)
            
            nbnd_lw = lookup_lw.n_bnd
            ngpt_lw = lookup_lw.n_gpt
            ngas = lookup_lw.n_gases

            if optics == :all
                ds_lw_cld = rrtmgp_artifact("lookup_tables", "cloudysky_lw.nc")
                lookup_lw_cld = RRTMGP.LookUpTables.LookUpCld(
                    ds_lw_cld,
                    Int,
                    FT,
                    DA,
                    !use_pade_method,
                )
                close(ds_lw_cld)
                lookups = (; lookups..., lookup_lw_cld)
            end
        end

        src_lw = 
            RRTMGP.Sources.source_func_longwave(FT, ncol, nlay, op_symb, DA)
        flux_lw = RRTMGP.Fluxes.FluxLW(ncol, nlay, FT, DA)
        fluxb_lw =
            optics == :gray ? nothing : RRTMGP.Fluxes.FluxLW(ncol, nlay, FT, DA)
        record_array!(rec, flux_lw.flux_up, "up_lw_flux", :lev, coords)
        record_array!(rec, flux_lw.flux_dn, "dn_lw_flux", :lev, coords)
        record_array!(rec, flux_lw.flux_net, "lw_flux", :lev, coords)
        if optics == :all_with_clear
            flux_lw_clear =
                RRTMGP.Fluxes.FluxLW(ncol, nlay + 1 - coords.nlay_ghost, FT, DA)
            push!(rec, (:clear_up_lw_flux, flux_lw_clear.flux_up))
            push!(rec, (:clear_dn_lw_flux, flux_lw_clear.flux_dn))
            push!(rec, (:clear_lw_flux, flux_lw_clear.flux_net))
        end

        sfc_emis = DA{FT}(undef, nbnd_lw, ncol)
        set_and_record_array!(sfc_emis, "surface_emissivity", :NA, tuple...)
        if :top_of_atmosphere_dn_lw_flux in keys(init_dict)
            inc_flux = DA{FT}(undef, ncol, ngpt_lw)
            set_and_record_array!(
                transpose(inc_flux),
                "top_of_atmosphere_dn_lw_flux",
                :NA,
                tuple...,
            )
        else
            inc_flux = nothing
        end
        bcs_lw = RRTMGP.BCs.LwBCs(sfc_emis, inc_flux)
    end

    if disable_shortwave
        src_sw = flux_sw = fluxb_sw = bcs_sw = nothing
    else
        if optics == :gray
            nbnd_sw = 1
            ngpt_sw = 1
        else
            ds_sw = rrtmgp_artifact("lookup_tables", "clearsky_sw.nc")
            lookup_sw, idx_gases =
                RRTMGP.LookUpTables.LookUpSW(ds_sw, Int, FT, DA)
            close(ds_sw)
            lookups = (; lookups..., lookup_sw, idx_gases)

            nbnd_sw = lookup_sw.n_bnd
            ngpt_sw = lookup_sw.n_gpt
            ngas = lookup_sw.n_gases
            
            if optics == :all
                ds_sw_cld = rrtmgp_artifact("lookup_tables", "cloudysky_sw.nc")
                lookup_sw_cld = RRTMGP.LookUpTables.LookUpCld(
                    ds_sw_cld,
                    Int,
                    FT,
                    DA,
                    !use_pade_method,
                )
                close(ds_sw_cld)
                lookups = (; lookups..., lookup_sw_cld)
            end
        end

        src_sw = 
            RRTMGP.Sources.source_func_shortwave(FT, ncol, nlay, op_symb, DA)
        flux_sw = RRTMGP.Fluxes.FluxSW(ncol, nlay, FT, DA)
        fluxb_sw =
            optics == :gray ? nothing : RRTMGP.Fluxes.FluxSW(ncol, nlay, FT, DA)
        record_array!(rec, flux_sw.flux_up, "up_sw_flux", :lev, coords)
        record_array!(rec, flux_sw.flux_dn, "dn_sw_flux", :lev, coords)
        record_array!(rec, flux_sw.flux_net, "sw_flux", :lev, coords)
        record_array!(rec, flux_sw.flux_dn_dir, "dir_dn_sw_flux", :lev, coords)
        if optics == :all_with_clear
            flux_lw_clear =
                RRTMGP.Fluxes.FluxSW(ncol, nlay + 1 - coords.nlay_ghost, FT, DA)
            push!(rec, (:clear_up_sw_flux, flux_sw_clear.flux_up))
            push!(rec, (:clear_dn_sw_flux, flux_sw_clear.flux_dn))
            push!(rec, (:clear_sw_flux, flux_sw_clear.flux_net))
            push!(rec, (:clear_dir_dn_sw_flux, flux_sw_clear.flux_dn_dir))
        end

        zenith = DA{FT}(undef, ncol)
        toa_flux = DA{FT}(undef, ncol)
        sfc_alb_direct = DA{FT}(undef, nbnd_sw, ncol)
        sfc_alb_diffuse = DA{FT}(undef, nbnd_sw, ncol)
        set_and_record_array!(zenith, "solar_zenith_angle", :NA, tuple...)
        set_and_record_array!(
            toa_flux,
            "weighted_irradiance",
            :NA,
            tuple...,
        )
        set_and_record_array!(
            sfc_alb_direct,
            "dir_sw_surface_albedo",
            :NA,
            tuple...,
        )
        set_and_record_array!(
            sfc_alb_diffuse,
            "dif_sw_surface_albedo",
            :NA,
            tuple...,
        )
        if :top_of_atmosphere_dif_dn_sw_flux in keys(init_dict)
            @warn string(
                "incoming diffuse shortwave fluxes are not yet implemented in ",
                "RRTMGP.jl; the value of keyword argument ",
                "top_of_atmosphere_dif_dn_sw_flux will be ignored",
            )
            inc_flux_diffuse = DA{FT}(undef, ncol, ngpt_sw)
            set_and_record_array!(
                transpose(inc_flux_diffuse),
                "top_of_atmosphere_dif_dn_sw_flux",
                :NA,
                tuple...,
            )
        else
            inc_flux_diffuse = nothing
        end
        bcs_sw = RRTMGP.BCs.SwBCs(
            zenith,
            toa_flux,
            sfc_alb_direct,
            inc_flux_diffuse,
            sfc_alb_diffuse,
        )
    end

    if disable_longwave
        record_array!(rec, flux_sw.flux_net, "flux", :lev, coords)
        if optics == :all_with_clear
            push!(rec, (:clear_flux, flux_sw_clear.flux_net))
        end
    elseif disable_shortwave
        record_array!(rec, flux_lw.flux_net, "flux", :lev, coords)
        if optics == :all_with_clear
            push!(rec, (:clear_flux, flux_lw_clear.flux_net))
        end
    else
        nlev_domain = nlay + 1 - coords.nlay_ghost
        push!(rec, (:flux, DA{FT}(undef, nlev_domain, ncol)))
        if optics == :all_with_clear
            push!(rec, (:clear_flux, DA{FT}(undef, nlev_domain, ncol)))
        end
        if optics != :gray
            @assert lookup_lw.n_gases == lookup_sw.n_gases
            @assert lookup_lw.p_ref_min == lookup_sw.p_ref_min
        end
    end

    p_lay = DA{FT}(undef, nlay, ncol)
    p_lev = DA{FT}(undef, nlay + 1, ncol)
    t_lay = DA{FT}(undef, nlay, ncol)
    t_lev = DA{FT}(undef, nlay + 1, ncol)
    implicit_lays =
        !(:pressure in keys(init_dict) && :temperature in keys(init_dict))
    implicit_levs =
        !(:level_pressure in keys(init_dict) && :level_temperature in keys(init_dict))
    if !implicit_lays
        set_and_record_array!(p_lay, "pressure", :lay, tuple...)
        set_and_record_array!(t_lay, "temperature", :lay, tuple...)
    end
    if !implicit_levs
        set_and_record_array!(p_lev, "level_pressure", :lev, tuple...)
        set_and_record_array!(t_lev, "level_temperature", :lev, tuple...)
    end
    t_sfc = DA{FT}(undef, ncol)
    set_and_record_array!(t_sfc, "surface_temperature", :NA, tuple...)

    if optics == :gray
        d0 = DA{FT}(undef, ncol)
        set_and_record_array!(d0, "optical_thickness_parameter", :NA, tuple...)
        if !(:lapse_rate in keys(init_dict))
            throw(UndefKeywordError(:lapse_rate))
        end
        α = pop!(init_dict, :lapse_rate)
        if !(α isa Real)
            throw(ArgumentError("keyword argument lapse_rate must be a Real"))
        end
        as = RRTMGP.AtmosphericStates.GrayAtmosphericState(
            p_lay,
            p_lev,
            t_lay,
            t_lev,
            DA{FT}(undef, nlay + 1, ncol), # TODO: z_lev required but never used
            t_sfc,
            α,
            d0,
            nlay,
            ncol,
        )
    else
        if !(:latitude in keys(init_dict))
            lon = lat = nothing
        else
            lon = DA{FT}(undef, ncol)
            lat = DA{FT}(undef, ncol)
            set_and_record_array!(lat, "latitude", :NA, tuple...)
        end

        vmr_str = "volume_mixing_ratio_"
        gas_names = filter(
            gas_name -> !(gas_name in ("h2o", "h2o_frgn", "h2o_self", "o3")),
            keys(idx_gases),
        )
        use_global_means = all(
            gas_name ->
                init_dict[Symbol(vmr_str * gas_name)] isa Real &&
                !(Symbol("extension_" * vmr_str * gas_name) in keys(init_dict)),
            gas_names,
        )
        # TODO: This gives the wrong types for CUDA 3.4 and above.
        # vmr = RRTMGP.Vmrs.init_vmr(ngas, nlay, ncol, FT, DA; gm = use_global_means)
        if use_global_means
            vmr = RRTMGP.Vmrs.VmrGM(
                DA{FT}(undef, nlay, ncol),
                DA{FT}(undef, nlay, ncol),
                DA{FT}(undef, ngas),
            )
            vmr.vmr .= zero(FT)
            set_and_record_array!(vmr.vmr_h2o, vmr_str * "h2o", :lay, tuple...)
            set_and_record_array!(vmr.vmr_o3, vmr_str * "o3", :lay, tuple...)
            for gas_name in gas_names
                set_and_record_array!(
                    @view(vmr.vmr[idx_gases[gas_name]]),
                    vmr_str * gas_name,
                    :NA,
                    tuple...,
                )
            end
        else
            vmr = RRTMGP.Vmrs.Vmr(DA{FT}(undef, nlay, ncol, ngas))
            for gas_name in ["h2o", "o3", gas_names...]
                set_and_record_array!(
                    @view(vmr.vmr[:, :, idx_gases[gas_name]]),
                    vmr_str * gas_name,
                    :lay,
                    tuple...,
                )
            end
        end

        if optics == :clear
            cld_r_eff_liq = cld_r_eff_ice = nothing
            cld_path_liq = cld_path_ice = cld_mask = nothing
            ice_rgh = 1
        else
            cld_r_eff_liq = DA{FT}(undef, nlay, ncol)
            cld_r_eff_ice = DA{FT}(undef, nlay, ncol)
            cld_path_liq = DA{FT}(undef, nlay, ncol)
            cld_path_ice = DA{FT}(undef, nlay, ncol)
            cld_mask = DA{Bool}(undef, nlay, ncol)
            set_and_record_array!(
                cld_r_eff_liq,
                "cloud_liquid_effective_radius",
                :lay,
                tuple...,
            )
            set_and_record_array!(
                cld_r_eff_ice,
                "cloud_ice_effective_radius",
                :lay,
                tuple...,
            )
            set_and_record_array!(
                cld_path_liq,
                "cloud_liquid_water_path",
                :lay,
                tuple...,
            )
            set_and_record_array!(
                cld_path_ice,
                "cloud_ice_water_path",
                :lay,
                tuple...,
            )
            set_and_record_array!(cld_mask, "cloud_mask", :lay, tuple...)
            if !(:ice_roughness in keys(init_dict))
                throw(UndefKeywordError(:ice_roughness))
            end
            ice_rgh = pop!(init_dict, :ice_roughness)
            if !(ice_rgh in (1, 2, 3))
                throw(ArgumentError(
                    "keyword argument ice_roughness must be 1, 2, or 3",
                ))
            end
        end

        as = RRTMGP.AtmosphericStates.AtmosphericState(
            lon, # TODO: lon required but never used
            lat,
            p_lay,
            p_lev,
            t_lay,
            t_lev,
            t_sfc,
            DA{FT}(undef, nlay, ncol),
            vmr,
            cld_r_eff_liq,
            cld_r_eff_ice,
            cld_path_liq,
            cld_path_ice,
            cld_mask,
            ice_rgh,
            nlay,
            ncol,
            ngas,
        )
    end

    if length(init_dict) > 0
        throw(ArgumentError(string(
            "got unexpected keyword argument",
            length(init_dict) == 1 ? " " : "s ",
            join(
                keys(init_dict),
                ", ",
                length(init_dict) == 2 ? " and " : ", and ",
            ),
        )))
    end

    if implicit_levs
        tuple =
            (level_computation, p_lay, p_lev, t_lay, t_lev, coords.domain.zs.lay, coords.domain.zs.lev, param_set)
        bottom_tuple = (
            use_ideal_coefs_for_bottom_level ? :ideal_coefs : level_computation,
            tuple[2:end]...,
        )

        if extension == 0
            i = nlay - coords.nlay_ghost
            level_olators = (level_olator(false, i + 1, i, i - 1, tuple...),)
        else
            i = nlay - coords.nlay_ghost + 1
            level_olators = ()
            
            i2 = nlay - coords.nlay_boundary
            # TODO: Will need another level_computation here if extension
            #       coordinates are not provided.
            level_olator(true, i + 1:i2, i:i2 - 1, i + 1:i2, tuple...)()
            level_olator(false, i2 + 1, i2, i2 - 1, tuple...)()
        end

        level_olators = (
            level_olator(true, 2:i, 1:i - 1, 2:i, tuple...),
            level_olator(false, 1, 1, 2, bottom_tuple...),
            level_olators...,
        )
    else
        if implicit_lays
            tuple =
                (level_computation, p_lev, p_lay, t_lev, t_lay, coords.domain.zs.lev, coords.domain.zs.lay, param_set)
            i = nlay - coords.nlay_ghost
            if extension != 0
                i2 = nlay - coords.nlay_boundary
                level_olator(true, i + 1:i2, i + 1:i2, i + 2:i2 + 1, tuple...)()
            end
            level_olators = (
                level_olator(true, 1:i, 1:i, 2:i + 1, tuple...),
            )
        else
            level_olators = ()
        end
        if use_ideal_coefs_for_bottom_level
            @warn string(
                "the bottom level will not be automatically computed when ",
                "`level_computation = :none`",
            )
        end
    end

    if add_isothermal_boundary_layer && extension != 0
        compute_boundary_layer!(as, lookups)
    end

    op_type = use_one_scalar ? RRTMGP.Optics.OneScalar : RRTMGP.Optics.TwoStream
    solver = RRTMGP.RTE.Solver(
        as,
        op_type(FT, ncol, nlay, DA),
        src_lw,
        src_sw,
        bcs_lw,
        bcs_sw,
        fluxb_lw,
        fluxb_sw,
        flux_lw,
        flux_sw,
    )

    mutable_values = (; rec...)
    return RRTMGPModel{
        optics,
        disable_longwave,
        disable_shortwave,
        extension == 0 ? add_isothermal_boundary_layer : false,
        typeof(solver),
        typeof(lookups),
        typeof(param_set),
        typeof(level_olators),
        typeof(mutable_values),
    }(
        solver,
        lookups,
        param_set,
        max_threads,
        level_olators,
        mutable_values,
    )
end

compute_boundary_layer!(::RRTMGPModel{Optics, No_LW, No_SW, false}) where
    {Optics, No_LW, No_SW} = nothing
compute_boundary_layer!(model::RRTMGPModel) =
    compute_boundary_layer!(model.solver.as, model.lookups)
function compute_boundary_layer!(as, lookups)
    p_min = get_p_min(as, lookups)
    as.p_lay[end, :] .= (as.p_lev[end - 1, :] .+ p_min) ./ 2
    as.p_lev[end, :] .= p_min
    as.t_lay[end, :] .= as.t_lev[end - 1, :]
    as.t_lev[end, :] .= as.t_lev[end - 1, :]
    compute_boundary_layer_vmr!(as)
end
get_p_min(as::RRTMGP.AtmosphericStates.GrayAtmosphericState, lookups) =
    zero(eltype(as.p_lay))
get_p_min(as::RRTMGP.AtmosphericStates.AtmosphericState, lookups) =
    lookups[1].p_ref_min
compute_boundary_layer_vmr!(as::RRTMGP.AtmosphericStates.GrayAtmosphericState) =
    nothing
compute_boundary_layer_vmr!(as::RRTMGP.AtmosphericStates.AtmosphericState) =
    compute_boundary_layer_vmr!(as.vmr)
function compute_boundary_layer_vmr!(vmr::RRTMGP.Vmrs.VmrGM)
    vmr.vmr_h2o[end, :] .= vmr.vmr_h2o[end - 1, :]
    vmr.vmr_o3[end, :] .= vmr.vmr_o3[end - 1, :]
end
function compute_boundary_layer_vmr!(vmr::RRTMGP.Vmrs.Vmr)
    vmr.vmr[end, :, :] .= vmr.vmr[end - 1, :, :]
end

# TODO: If the extension profile doesn't change, we don't need to recompute
#       `col_dry` for it.
compute_concentrations!(::RRTMGPModel{:gray}) = nothing
compute_concentrations!(model::RRTMGPModel) = RRTMGP.Optics.compute_col_dry!(
    model.solver.as.p_lev,
    model.solver.as.col_dry,
    model.param_set,
    get_vmr_h2o(model.solver.as.vmr, model.lookups.idx_gases),
    model.solver.as.lat,
    model.max_threads,
)
get_vmr_h2o(vmr::RRTMGP.Vmrs.VmrGM, idx_gases) = vmr.vmr_h2o
get_vmr_h2o(vmr::RRTMGP.Vmrs.Vmr, idx_gases) =
    @view vmr.vmr[:, :, idx_gases["h2o"]]

compute_lw_fluxes!(::RRTMGPModel) = nothing
compute_lw_fluxes!(model::RRTMGPModel{:gray, false}) =
    RRTMGP.RTESolver.solve_lw!(
        model.solver,
        model.max_threads,
    )
compute_lw_fluxes!(model::RRTMGPModel{:clear, false}) =
    RRTMGP.RTESolver.solve_lw!(
        model.solver,
        model.max_threads,
        model.lookups.lookup_lw,
    )
compute_lw_fluxes!(model::RRTMGPModel{:all, false}) =
    RRTMGP.RTESolver.solve_lw!(
        model.solver,
        model.max_threads,
        model.lookups.lookup_lw,
        model.lookups.lookup_lw_cld,
    )
function compute_lw_fluxes!(model::RRTMGPModel{:all_with_clear, false})
    RRTMGP.RTESolver.solve_lw!(
        model.solver,
        model.max_threads,
        model.lookups.lookup_lw,
    )
    model.clear_up_lw_flux .= model.up_lw_flux
    model.clear_dn_lw_flux .= model.dn_lw_flux
    model.clear_lw_flux .= model.lw_flux
    RRTMGP.RTESolver.solve_lw!(
        model.solver,
        model.max_threads,
        model.lookups.lookup_lw,
        model.lookups.lookup_lw_cld,
    )
end

import RRTMGP.Optics: compute_optical_props_kernel!
function compute_optical_props_kernel!(
    op::RRTMGP.Optics.AbstractOpticalProps{FT},
    as::RRTMGP.AtmosphericStates.GrayAtmosphericState{FT},
    glaycol,
    source::RRTMGP.Sources.AbstractSourceLW{FT},
) where {FT<:AbstractFloat}
    compute_optical_props_kernel_lw!(op, as, glaycol)     # computing optical thickness
    RRTMGP.Optics.compute_sources_gray_kernel!(source, as, glaycol) # computing Planck sources
end
function compute_optical_props_kernel_lw!(
    op::RRTMGP.Optics.AbstractOpticalProps{FT},
    as::RRTMGP.AtmosphericStates.GrayAtmosphericState{FT},
    glaycol,
) where {FT<:AbstractFloat}
    # setting references
    glay, gcol = glaycol
    @unpack p_lay, p_lev, d0 = as
    @inbounds p0 = p_lev[1, gcol]

    f = FT(0.2)
    α = FT(1)
    τ0_lw = d0
    @inbounds op.τ[glaycol...] =
        α * τ0_lw[gcol] * (f / p0 + FT(4) * (FT(1) - f) / p0 * (p_lay[glaycol...] / p0)^3) * (p_lev[glaycol...] - p_lev[glay+1, gcol])

    if op isa RRTMGP.Optics.TwoStream
        op.ssa[glaycol...] = FT(0)
        op.g[glaycol...] = FT(0)
    end
end
function compute_optical_props_kernel!(
    op::RRTMGP.Optics.AbstractOpticalProps{FT},
    as::RRTMGP.AtmosphericStates.GrayAtmosphericState{FT},
    glaycol,
) where {FT<:AbstractFloat}
    # setting references
    glay, gcol = glaycol
    @unpack p_lay, p_lev = as
    @inbounds p0 = p_lev[1, gcol]

    τ0_sw = FT(0.22)
    @inbounds op.τ[glaycol...] =
        FT(2) * τ0_sw * (p_lay[glaycol...] / p0) / p0 * (p_lev[glaycol...] - p_lev[glay+1, gcol])

    if op isa RRTMGP.Optics.TwoStream
        op.ssa[glaycol...] = FT(0)
        op.g[glaycol...] = FT(0)
    end
end

compute_sw_fluxes!(::RRTMGPModel) = nothing
compute_sw_fluxes!(model::RRTMGPModel{:gray, No_LW, false}) where {No_LW} =
    RRTMGP.RTESolver.solve_sw!(
        model.solver,
        model.max_threads,
    )
compute_sw_fluxes!(model::RRTMGPModel{:clear, No_LW, false}) where {No_LW} =
    RRTMGP.RTESolver.solve_sw!(
        model.solver,
        model.max_threads,
        model.lookups.lookup_sw,
    )
compute_sw_fluxes!(model::RRTMGPModel{:all, No_LW, false}) where {No_LW} =
    RRTMGP.RTESolver.solve_sw!(
        model.solver,
        model.max_threads,
        model.lookups.lookup_sw,
        model.lookups.lookup_sw_cld,
    )
function compute_sw_fluxes!(
    model::RRTMGPModel{:all_with_clear, No_LW, false},
) where {No_LW}
    RRTMGP.RTESolver.solve_sw!(
        model.solver,
        model.max_threads,
        model.lookups.lookup_sw,
    )
    model.clear_up_sw_flux .= model.up_sw_flux
    model.clear_dn_sw_flux .= model.dn_sw_flux
    model.clear_sw_flux .= model.sw_flux
    model.clear_dir_dn_sw_flux .= model.dir_dn_sw_flux
    RRTMGP.RTESolver.solve_sw!(
        model.solver,
        model.max_threads,
        model.lookups.lookup_sw,
        model.lookups.lookup_sw_cld,
    )
end

compute_net_flux!(model::RRTMGPModel) = model.flux
compute_net_flux!(model::RRTMGPModel{Optics, false, false}) where {Optics} =
    model.flux .= model.lw_flux .+ model.sw_flux
function compute_net_flux!(model::RRTMGPModel{:all_with_clear, false, false})
    model.clear_flux .= model.clear_lw_flux .+ model.clear_sw_flux
    model.flux .= model.lw_flux .+ model.sw_flux
end

function compute_fluxes!(model::RRTMGPModel)
    for level_olator in model.level_olators
        level_olator()
    end
    model.solver.as.p_lay .=
        max.(model.solver.as.p_lay, get_p_min(model.solver.as, model.lookups))
    model.solver.as.p_lev .=
        max.(model.solver.as.p_lev, get_p_min(model.solver.as, model.lookups))
    # model.solver.as.t_lev[1, :] .= model.solver.as.t_sfc
    # model.solver.as.p_lev[end, :] .= get_p_min(model.solver.as, model.lookups)
    # model.solver.as.t_lev[end, :] .= model.solver.as.t_lay[end, :]
    compute_boundary_layer!(model)
    compute_concentrations!(model)
    compute_lw_fluxes!(model)
    compute_sw_fluxes!(model)
    return compute_net_flux!(model)
end