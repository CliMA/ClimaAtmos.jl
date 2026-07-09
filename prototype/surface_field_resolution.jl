#=
Prototype: field-level surface resolution — the hardest case surfaced by
bomex_type_dispatch.jl. See TYPE_DISPATCH_MIGRATION.md.

`AtmosSurface` is NOT resolvable as a single component. Each of its four fields
has its own policy, and config keys act as *gates* that change the resolution
mode:

  - temperature       : config GATE `prognostic_surface`
                          "SlabOceanSST" -> SlabOceanTemperature   (config wins)
                          "PrescribedSST" -> setup.temperature, else default
  - flux_scheme       : setup wins; else config. The config value can itself be
                        `nothing` ("PrescribedSurface" sentinel) -> a NULLABLE
                        resolved field.
  - boundary_overrides: setup.overrides, else default
  - surface_albedo    : config-only registry (`albedo_model`)

Two design conclusions this prototype validates:

  (1) Resolution must be PER FIELD, with some config keys selecting the *mode*
      (gates), not just supplying a value.
  (2) `@something` is insufficient: a field can legitimately resolve to
      `nothing`, and the Setup layer's `nothing` means "fall through" while the
      config layer's `nothing` is a real value. We need an explicit `UNSET`
      sentinel and per-layer null semantics.

Strategy: build surface via the proposed design for three scenarios and assert
field-by-field type equality against the real `AtmosSurface`.

Run:
    julia --project=.buildkite prototype/surface_field_resolution.jl
=#

import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaAtmos.Setups as Setups
import ClimaAtmos.Parameters as CAP
const SC = CA.SurfaceConditions

const FT = Float32
const BOMEX_CFG = joinpath(
    pkgdir(CA),
    "config",
    "model_configs",
    "prognostic_edmfx_bomex_column.yml",
)

# ---------------------------------------------------------------------------
# Registry (string -> builder(ctx)), as in bomex_type_dispatch.jl.
# ---------------------------------------------------------------------------
const REGISTRIES = Dict{Symbol, Dict{String, Any}}()
_registry(name) = get!(Dict{String, Any}, REGISTRIES, name)
register!(name, key, builder) = (_registry(name)[key] = builder)
from_registry(name, key, ctx = (;)) = _registry(name)[key](ctx)

# `surface_setup` (flux scheme) — an explicit registry instead of the current
# `getproperty(SurfaceConditions, Symbol(name))` reflection.
register!(
    :flux_scheme,
    "DefaultExchangeCoefficients",
    ctx -> SC.DefaultExchangeCoefficients()(ctx.params),
)
register!(:flux_scheme, "DefaultMoninObukhov", ctx -> SC.DefaultMoninObukhov()(ctx.params))

# `albedo_model`
register!(
    :albedo,
    "ConstantAlbedo",
    ctx -> CA.ConstantAlbedo{FT}(; α = ctx.params.idealized_ocean_albedo),
)
register!(:albedo, "CouplerAlbedo", ctx -> CA.CouplerAlbedo())
register!(
    :albedo,
    "RegressionFunctionAlbedo",
    ctx -> CA.RegressionFunctionAlbedo{FT}(; n = ctx.params.water_refractive_index),
)

# ---------------------------------------------------------------------------
# Field-level resolution with an explicit UNSET sentinel.
# Each layer is either UNSET ("not provided") or a value (which may be `nothing`,
# a legitimate resolved value). The first non-UNSET layer wins.
# ---------------------------------------------------------------------------
struct Unset end
const UNSET = Unset()
function resolve_field(layers...)
    for l in layers
        l === UNSET || return l   # NOTE: `nothing` is a valid resolved value
    end
    error("no value (not even UNSET fallthrough) provided for field")
end
# A Setup hook returns `nothing` to mean "fall through"; convert that to UNSET.
setup_layer(v) = v === nothing ? UNSET : v

# ---------------------------------------------------------------------------
# Proposed surface construction.
# ---------------------------------------------------------------------------
function build_surface(pa, params, setup; overrides = (;))
    sfc =
        isnothing(setup) ?
        (; flux_scheme = nothing, temperature = nothing, overrides = nothing) :
        Setups.surface_condition(setup, params)

    # temperature: config GATE selects the mode
    temperature = resolve_field(
        get(overrides, :temperature, UNSET),
        if pa["prognostic_surface"] == "SlabOceanSST"
            SC.SlabOceanTemperature{FT}()                      # config wins
        else                                                   # PrescribedSST
            @something(sfc.temperature, Setups.surface_temperature_model(setup))
        end,
    )

    # flux_scheme: setup > config; config value may itself be `nothing`.
    flux_scheme = resolve_field(
        get(overrides, :flux_scheme, UNSET),
        setup_layer(sfc.flux_scheme),                          # setup nothing = fall through
        pa["surface_setup"] == "PrescribedSurface" ? nothing : # config nothing = real value
        from_registry(:flux_scheme, pa["surface_setup"], (; params)),
    )

    boundary_overrides = resolve_field(
        setup_layer(sfc.overrides),
        SC.SurfaceBoundaryOverrides(),
    )

    # surface_albedo: config-only
    surface_albedo = from_registry(:albedo, pa["albedo_model"], (; params))

    return CA.AtmosSurface(;
        flux_scheme,
        temperature,
        boundary_overrides,
        surface_albedo,
    )
end

# ---------------------------------------------------------------------------
# Scenario runner: compare proposed build to the real AtmosSurface.
# ---------------------------------------------------------------------------
sametype(a, b) = typeof(a) == typeof(b)
function check_surface(name, config_dict)
    config = CA.AtmosConfig(config_dict; job_id = "surf_$(name)")
    params = CA.ClimaAtmosParameters(config)
    setup = CA.get_setup_type(
        config.parsed_args,
        CAP.thermodynamics_params(params),
    )
    ref = CA.AtmosSurface(config, params, FT; setup_type = setup)
    new = build_surface(config.parsed_args, params, setup)

    fields = (:flux_scheme, :temperature, :boundary_overrides, :surface_albedo)
    oks = map(fields) do f
        ok = sametype(getfield(new, f), getfield(ref, f))
        ok ||
            @warn "MISMATCH" scenario = name field = f ref = typeof(getfield(ref, f)) got =
                typeof(getfield(new, f))
        ok
    end
    @info "scenario" name reproduced = all(oks)
    @assert all(oks)
end

function main()
    # A: sphere defaults — setup provides no surface pieces; flux from
    #    `surface_setup` registry, temperature from the setup's default model.
    check_surface("sphere_default", Dict("config" => "sphere"))

    # B: BOMEX column — the Setup supplies flux_scheme/temperature/overrides;
    #    albedo still from config.
    check_surface("bomex_column", BOMEX_CFG)

    # C: coupler-like — `prognostic_surface: SlabOceanSST` gate makes the config
    #    temperature win; `surface_setup: PrescribedSurface` makes flux resolve
    #    to `nothing`; `albedo_model: CouplerAlbedo`.
    check_surface(
        "coupler",
        Dict(
            "config" => "sphere",
            "prognostic_surface" => "SlabOceanSST",
            "surface_setup" => "PrescribedSurface",
            "albedo_model" => "CouplerAlbedo",
        ),
    )

    # Sentinel/precedence unit checks for resolve_field:
    @assert resolve_field(UNSET, UNSET, 42) == 42
    @assert resolve_field(UNSET, 7, 42) == 7
    @assert resolve_field(UNSET, nothing, 42) === nothing   # nothing is a real value
    @assert resolve_field(setup_layer(nothing), 42) == 42   # setup nothing => fall through

    @info "PROTOTYPE PASSED: field-level surface resolution reproduces AtmosSurface"
end

main()
