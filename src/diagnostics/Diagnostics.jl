module Diagnostics

import Dates: Month, DateTime, Period

import ClimaComms

import LinearAlgebra: dot

import ClimaCore:
    Fields, Geometry, InputOutput, Meshes, Spaces, Operators, Domains, Grids
import ClimaCore.Utilities: half
import Thermodynamics as TD

import ..AtmosModel
import ..Φ
import ..AtmosCallback
import ..EveryNSteps

import ..Parameters as CAP

import ..unit_basis_vector_data

# moisture_model
import ..DryModel
import ..EquilMoistModel
import ..NonEquilMoistModel

# precip_model
import ..NoPrecipitation
import ..Microphysics0Moment
import ..Microphysics1Moment

# radiation
import ClimaAtmos.RRTMGPInterface as RRTMGPI

# vert_diff
import ..VerticalDiffusion
import ..FriersonDiffusion
import ..DecayWithHeightDiffusion

# turbconv_model
import ..PrognosticEDMFX
import ..DiagnosticEDMFX

# surface_model
import ..PrognosticSurfaceTemperature

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
