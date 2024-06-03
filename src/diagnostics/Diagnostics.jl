module Diagnostics

import ClimaComms

import LinearAlgebra: dot

import ClimaCore:
    Fields, Geometry, InputOutput, Meshes, Spaces, Operators, Domains, Grids
import ClimaCore.Utilities: half
import Thermodynamics as TD

import ..AtmosModel
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

# turbconv_model
import ..SmagorinskyLilly
import ..PrognosticEDMFX
import ..DiagnosticEDMFX

# functions used to calculate diagnostics
import ..draft_area
import ..compute_gm_mixing_length!

# We need the abbreviations for symbols like curl, grad, and so on
include(joinpath("..", "utils", "abbreviations.jl"))

import ClimaDiagnostics:
    DiagnosticVariable, ScheduledDiagnostic, average_pre_output_hook!

import ClimaDiagnostics.DiagnosticVariables: descriptive_short_name

import ClimaDiagnostics.Schedules: EveryStepSchedule, EveryDtSchedule

import ClimaDiagnostics.Writers:
    HDF5Writer,
    NetCDFWriter,
    write_field!,
    LevelsMethod,
    FakePressureLevelsMethod

include("diagnostic.jl")

end
