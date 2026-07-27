#=
Column dataset tests for ClimaAtmos.jl

These tests cover the ColumnDatasets subsystem: format conformance checks,
file probing, reading ClimaColumn (z, time) files onto a column space, and the
ForcingFromFile setup built on top of the dataset interface.
=#

using Test
using ClimaComms
ClimaComms.@import_required_backends
import Dates
using Random
Random.seed!(1234)
import ClimaAtmos as CA
import ClimaAtmos.ColumnDatasets as CD
using NCDatasets
import ClimaCore
import Interpolations as Intp
import ClimaUtilities.TimeVaryingInputs: evaluate!

include("test_helpers.jl")

"""
Generate one ClimaColumn forcing file from the mock ERA5 inputs.
"""
function generate_test_forcing_file(FT)
    parsed_args = Dict(
        "start_date" => "20000506",
        "site_latitude" => 0.0,
        "site_longitude" => 0.0,
        "t_end" => "5hours",
        "era5_diurnal_warming" => Nothing,
    )
    temporary_dir = mktempdir()
    forcing_file = CA.get_external_daily_forcing_file_path(
        parsed_args,
        data_dir = temporary_dir,
    )
    create_mock_era5_datasets(temporary_dir, parsed_args["start_date"], FT)
    CA.generate_external_forcing_file(
        parsed_args,
        forcing_file,
        FT,
        input_data_dir = temporary_dir,
    )
    return forcing_file
end

"""
Write a legacy 2x2 box forcing file with uniform column values; when
`with_surface_vars` is set, also write box-replicated 4D surface variables.
"""
function write_test_legacy_box_file(path, FT; with_surface_vars = false)
    nz, nt = 6, 4
    ds = NCDataset(path, "c")
    defDim(ds, "x", 2)
    defDim(ds, "y", 2)
    defDim(ds, "z", nz)
    defDim(ds, "time", nt)
    defVar(ds, "x", FT, ("x",))
    ds["x"][:] = [0.0, 1.0]
    defVar(ds, "y", FT, ("y",))
    ds["y"][:] = [0.0, 1.0]
    defVar(ds, "z", FT, ("z",))
    ds["z"][:] = collect(range(100.0, 5000.0, nz))
    defVar(
        ds,
        "time",
        collect(0.0:(nt - 1)),
        ("time",),
        attrib = [
            "units" => "hours since 2000-05-06 00:00:00",
            "calendar" => "standard",
        ],
    )
    column_values = [
        ("ta", 280.0),
        ("ua", 1.0),
        ("va", 2.0),
        ("hus", 0.01),
        ("rho", 1.0),
        ("wa", -0.01),
    ]
    surface_values = [("ts", 275.0), ("hfls", 10.0), ("hfss", 5.0)]
    values = with_surface_vars ? [column_values; surface_values] : column_values
    for (name, value) in values
        defVar(ds, name, FT, ("x", "y", "z", "time"))
        ds[name][:, :, :, :] .= value
    end
    close(ds)
    return path
end

@testset "Format conformance and construction" begin
    FT = Float64
    temporary_dir = mktempdir()

    conforming_file = generate_test_forcing_file(FT)
    box_file = write_test_legacy_box_file(
        joinpath(temporary_dir, "legacy_box_forcing.nc"),
        FT,
    )

    # `is_conforming` distinguishes native ClimaColumn files from non-native
    # (box) files; this is the check that triggers on-demand regeneration of a
    # stale cached file.
    @test CD.ClimaColumnFiles.is_conforming(conforming_file)
    @test !CD.ClimaColumnFiles.is_conforming(box_file)

    # There is a single format: non-native and unrecognized files are a loud
    # error at construction (the reader assumes the native schema; `validate`
    # rejects everything else, rather than guessing).
    @test_throws ErrorException CD.ColumnDataset(box_file)
    unknown_file = joinpath(temporary_dir, "unknown.nc")
    NCDataset(unknown_file, "c") do ds
        defDim(ds, "lev", 3)
        defVar(ds, "lev", FT, ("lev",))
    end
    @test_throws ErrorException CD.ColumnDataset(unknown_file)

    # the probe captures availability and the schema metadata
    data = CD.ColumnDataset(conforming_file)
    @test data.format isa CD.ClimaColumnFile
    @test issetequal(data.column_vars, collect(CD.CANONICAL_COLUMN_VARS))
    @test issetequal(data.surface_vars, collect(CD.CANONICAL_SURFACE_VARS))
    @test isempty(CD.missing_forcing_variables(data))
    @test isnothing(CD.validate(data))

    # native files are validated during construction because readers assume
    # the canonical SI units without performing conversions
    invalid_units_file = generate_test_forcing_file(FT)
    NCDataset(invalid_units_file, "a") do ds
        ds["ta"].attrib["units"] = "degC"
    end
    @test_throws ErrorException CD.ColumnDataset(invalid_units_file)

    # A native forcing-only file may omit data for explicitly disabled
    # consumers. Enabled terms still require their complete input set.
    limited_file = joinpath(temporary_dir, "limited_forcing.nc")
    CD.ClimaColumnFiles.write_column_forcing_file(
        limited_file,
        FT;
        z = FT[100, 200],
        time = [
            Dates.DateTime(2000, 5, 6),
            Dates.DateTime(2000, 5, 6, 1),
        ],
        time_attrib = [
            "units" => "hours since 2000-05-06 00:00:00",
            "calendar" => "standard",
        ],
        column_vars = Dict(
            "tntha" => zeros(FT, 2, 2),
            "tnhusha" => zeros(FT, 2, 2),
        ),
        surface_vars = Dict{String, Vector{FT}}(),
        site_latitude = 0,
        site_longitude = 0,
    )
    limited_data = CD.ColumnDataset(limited_file)
    # The composed forcing's required column variables are checked against the
    # file; a horizontal-advection-only composition needs only tntha/tnhusha.
    hadv_only = (CA.HorizontalAdvection(),)
    hadv_vars = (:tntha, :tnhusha)
    @test isnothing(
        CD.require_forcing_variables(limited_data, hadv_vars, ()),
    )
    # ts/coszen/rsdt (surface, required by the resolved model) are absent.
    @test CD.missing_forcing_variables(
        limited_data,
        hadv_vars,
        (:ts, :coszen, :rsdt),
    ) == [:ts, :coszen, :rsdt]
    @test CA.ExternalDrivenTVForcing(
        limited_data;
        forcing = hadv_only,
    ).forcing === hadv_only
    # The default composition needs nudging/subsidence data the file lacks.
    default_vars =
        Tuple(union(CA.required_column_variables.(CA.default_forcing_terms())...))
    @test_throws ErrorException CD.require_forcing_variables(
        limited_data,
        default_vars,
        (),
    )
end

@testset "ClimaColumn files onto a column space" begin
    FT = Float64
    forcing_file = generate_test_forcing_file(FT)
    data = CD.ColumnDataset(forcing_file)
    start_date = Dates.DateTime(2000, 5, 6)

    grid = CA.ColumnGrid(FT; z_elem = 25, z_max = FT(20e3), z_stretch = false)
    center_space = ClimaCore.Spaces.CenterFiniteDifferenceSpace(grid)
    ᶜz = ClimaCore.Fields.coordinate_field(center_space).z

    column_inputs = CD.column_timevaryinginputs(
        data,
        CD.CANONICAL_COLUMN_VARS,
        center_space,
        start_date,
    )
    @test keys(column_inputs) == CD.CANONICAL_COLUMN_VARS

    dest = zero.(ᶜz)
    evaluate!(dest, column_inputs.ua, 0.0)
    @test all(parent(dest) .≈ 1)

    # the pure 1D (z, time) read reproduces direct interpolation of the file
    # profile onto the column, with flat extrapolation beyond the file range
    evaluate!(dest, column_inputs.ta, 0.0)
    z_file, ta_file = NCDataset(forcing_file) do ds
        (Float64.(ds["z"][:]), Float64.(ds["ta"][:, 1]))
    end
    ta_itp = Intp.extrapolate(
        Intp.interpolate((z_file,), ta_file, Intp.Gridded(Intp.Linear())),
        Intp.Flat(),
    )
    @test all(
        isapprox.(vec(parent(dest)), ta_itp.(vec(parent(ᶜz))), rtol = 1e-6),
    )

    # surface series are read as in-memory (time,) inputs
    surface_fields = CD.surface_timevaryinginputs(
        data,
        CD.CANONICAL_SURFACE_VARS,
        center_space,
        start_date,
    )
    @test keys(surface_fields) == CD.CANONICAL_SURFACE_VARS
    evaluate!(dest, surface_fields.ts, 0.0)
    @test all(parent(dest) .≈ 1)

    # the ForcingFromFile setup sources IC and models from the same handle
    setup = CA.Setups.ForcingFromFile(data, "20000506")
    @test setup.dataset === data
    @test CA.Setups.external_forcing(setup, FT) isa CA.ExternalDrivenTVForcing
    @test CA.Setups.external_forcing(setup, FT).dataset === data
    @test CA.Setups.insolation_model(setup) isa CA.ExternalTVInsolation
    @test CA.Setups.surface_temperature_model(setup) isa
          CA.SurfaceConditions.ExternalTemperature
    # the surface prescribes the file SST with interactive Monin-Obukhov fluxes
    params = CA.ClimaAtmosParameters(FT)
    surface = CA.Setups.surface_condition(setup, params)
    @test surface.flux_scheme isa CA.SurfaceConditions.MoninObukhov
    @test isnothing(surface.temperature)
    # the IC profiles reproduce the (uniform) file values
    @test setup.profiles.u(2000.0) ≈ 1
    @test ta_itp(2000.0) ≈ setup.profiles.T(2000.0)

    # a narrower forcing composition flows through to the forcing model
    advection_only = (CA.HorizontalAdvection(),)
    advection_setup = CA.Setups.ForcingFromFile(
        data,
        "20000506";
        forcing = advection_only,
    )
    @test CA.Setups.external_forcing(advection_setup, FT).forcing ===
          advection_only
end

@testset "Required-variable checks" begin
    FT = Float64
    # A native file carrying only initial-state, nudging, and subsidence
    # variables (no advection tendencies, no surface fluxes or insolation)
    # exercises the missing/required-variable logic without any legacy format.
    nz, nt = 6, 4
    partial = joinpath(mktempdir(), "partial_native.nc")
    CD.ClimaColumnFiles.write_column_forcing_file(
        partial,
        FT;
        z = collect(range(100.0, 5000.0, nz)),
        time = [Dates.DateTime(2000, 5, 6) + Dates.Hour(h) for h in 0:(nt - 1)],
        time_attrib = [
            "units" => "seconds since 1970-01-01T00:00:00",
            "calendar" => "proleptic_gregorian",
        ],
        column_vars = Dict(
            "ta" => fill(280.0, nz, nt),
            "hus" => fill(0.01, nz, nt),
            "ua" => fill(1.0, nz, nt),
            "va" => fill(2.0, nz, nt),
            "wa" => fill(-0.01, nz, nt),
            "rho" => fill(1.0, nz, nt),
        ),
        surface_vars = Dict("ts" => fill(275.0, nt)),
        site_latitude = 0.0,
        site_longitude = 0.0,
    )

    data = CD.ColumnDataset(partial)
    @test issetequal(data.column_vars, [:ta, :hus, :ua, :va, :wa])
    @test issetequal(data.surface_vars, [:ts])

    # single-argument form: everything the file is missing from the canonical
    # vocabulary (never silently skipped)
    @test issetequal(
        CD.missing_forcing_variables(data),
        [:tntha, :tnhusha, :tntva, :tnhusva, :hfls, :hfss, :coszen, :rsdt],
    )

    # a composition satisfied by the present variables is accepted ...
    limited_vars = Tuple(
        union(
            CA.required_column_variables.((
                CA.Nudging(:ta, :hus),
                CA.Nudging(:ua, :va),
                CA.Subsidence(),
            ))...,
        ),
    )
    @test isempty(CD.missing_forcing_variables(data, limited_vars, ()))
    @test isnothing(CD.require_forcing_variables(data, limited_vars, ()))

    # ... but the default composition needs advection tendencies, a loud error
    default_vars =
        Tuple(union(CA.required_column_variables.(CA.default_forcing_terms())...))
    @test_throws ErrorException CD.require_forcing_variables(data, default_vars, ())
end

@testset "Forcing-term composition" begin
    # construction validation is loud
    @test CA.Nudging(:ta, :hus) isa CA.Nudging
    @test_throws ErrorException CA.Nudging(:foo)          # unknown variable
    @test_throws ErrorException CA.Nudging(:ua)           # momentum needs :va too
    @test_throws ErrorException CA.Nudging(:ta, :ua, :va)  # mixed + DefaultTimescale
    @test CA.Nudging(:ta, :ua, :va; timescale = 3600.0) isa CA.Nudging  # explicit ok

    @test_throws ErrorException CA.validate_forcing_terms((
        CA.HorizontalAdvection(),
        CA.HorizontalAdvection(),
    ))
    @test_throws ErrorException CA.validate_forcing_terms((
        CA.Nudging(:ta),
        CA.Nudging(:ta; timescale = 100.0),  # :ta nudged twice
    ))

    # per-term required variables
    @test CA.required_column_variables(CA.HorizontalAdvection()) ==
          (:tntha, :tnhusha)
    @test CA.required_column_variables(CA.Subsidence()) == (:wa,)
    @test CA.required_column_variables(CA.Nudging(:ta, :hus)) == (:ta, :hus)

    # surface-variable requirements are derived from the model components,
    # not the forcing terms
    @test CA.required_surface_variables(
        CA.SurfaceConditions.ExternalTemperature(),
    ) == (:ts,)
    @test CA.required_surface_variables(CA.ExternalTVInsolation()) ==
          (:coszen, :rsdt)
    @test CA.required_surface_variables(
        CA.SurfaceConditions.SlabOceanTemperature{Float64}(),
    ) ==
          ()
end

@testset "ForcingFromFile surface and insolation seams" begin
    FT = Float64
    forcing_file = generate_test_forcing_file(FT)
    data = CD.ColumnDataset(forcing_file)

    # site_location reads the schema's site attributes
    @test CD.site_location(data) == (; latitude = 0.0, longitude = 0.0)

    # FileHeatFluxes reads hfls/hfss and returns a HeatFluxes closure
    fluxes = CA.SurfaceConditions.FileHeatFluxes(data, "20000506")
    hf = fluxes(0.0, FT)
    @test hf isa CA.SurfaceConditions.HeatFluxes
    @test isfinite(hf.shf) && isfinite(hf.lhf)

    # loud error when the file lacks hfls/hfss
    no_flux_file = joinpath(mktempdir(), "no_flux.nc")
    CD.ClimaColumnFiles.write_column_forcing_file(
        no_flux_file,
        FT;
        z = FT[100, 200],
        time = [Dates.DateTime(2000, 5, 6), Dates.DateTime(2000, 5, 6, 1)],
        time_attrib = [
            "units" => "hours since 2000-05-06 00:00:00",
            "calendar" => "standard",
        ],
        column_vars = Dict("tntha" => zeros(FT, 2, 2)),
        surface_vars = Dict{String, Vector{FT}}(),
        site_latitude = 0,
        site_longitude = 0,
    )
    @test_throws ErrorException CA.SurfaceConditions.FileHeatFluxes(
        CD.ColumnDataset(no_flux_file),
        "20000506",
    )

    # the insolation and flux_scheme kwargs are stored and returned by traits
    (lat, lon) = CD.site_location(data)
    setup = CA.Setups.ForcingFromFile(
        data,
        "20000506";
        flux_scheme = CA.SurfaceConditions.MoninObukhov(;
            z0 = FT(0.05),
            ustar = FT(0.28),
            fluxes = CA.SurfaceConditions.FileHeatFluxes(data, "20000506"),
        ),
        insolation = CA.TimeVaryingInsolation(;
            start_date = Dates.DateTime(2000, 5, 6),
            latitude = lat,
            longitude = lon,
        ),
    )
    params = CA.ClimaAtmosParameters(FT)
    scheme = CA.Setups.surface_condition(setup, params).flux_scheme
    @test scheme isa CA.SurfaceConditions.MoninObukhov
    @test !isnothing(scheme.fluxes)   # prescribed-flux closure plumbed through
    @test CA.Setups.insolation_model(setup) isa CA.TimeVaryingInsolation
end
