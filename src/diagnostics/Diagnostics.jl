module Diagnostics

import Dates: Month, Day, Hour, DateTime, Period

import LinearAlgebra: dot

import ClimaComms
import ClimaCore:
    Fields, Geometry, InputOutput, Meshes, Spaces, Operators, Domains, Grids
import ClimaCore.Utilities: half
import Thermodynamics as TD

# compute lazily to reduce allocations
import ..lazy

import ..AtmosModel
import ..AtmosWater
import ..AtmosRadiation
import ..AtmosTurbconv
import ..AtmosCallback
import ..EveryNSteps

import ..Parameters as CAP

import ..unit_basis_vector_data

# moisture_model
import ..DryModel
import ..EquilMoistModel
import ..NonEquilMoistModel

# microphysics_model
import ..NoPrecipitation
import ..Microphysics0Moment
import ..Microphysics1Moment
import ..Microphysics2Moment

# radiation
import ClimaAtmos.RRTMGPInterface as RRTMGPI

# vert_diff
import ..VerticalDiffusion
import ..DecayWithHeightDiffusion

# turbconv_model
import ..EDOnlyEDMFX
import ..PrognosticEDMFX
import ..DiagnosticEDMFX

# gravitywave_models
import ..NonOrographicGravityWave
import ..OrographicGravityWave

# surface_model
import ..SlabOceanSST

# functions used to calculate diagnostics
import ..draft_area
import ..compute_gm_mixing_length!
import ..horizontal_integral_at_boundary

# We need the abbreviations for symbols like curl, grad, and so on
include(joinpath("..", "utils", "abbreviations.jl"))

import ClimaDiagnostics

import ClimaDiagnostics:
    DiagnosticVariable, ScheduledDiagnostic, average_pre_output_hook!

import ClimaDiagnostics.DiagnosticVariables: descriptive_short_name

import ClimaDiagnostics.Schedules:
    EveryStepSchedule, EveryDtSchedule, EveryCalendarDtSchedule

import ClimaDiagnostics.Writers:
    DictWriter,
    HDF5Writer,
    NetCDFWriter,
    write_field!,
    LevelsMethod,
    FakePressureLevelsMethod

include("diagnostic.jl")

end
