# ============================================================================
# Microphysics Module
# ============================================================================
# Cloud and precipitation physics for ClimaAtmos.
#
# This module contains:
# - SGS quadrature infrastructure for subgrid-scale integration
# - Cloud fraction diagnostics and ML-based cloud fraction
# - Tendency limiters for numerical stability
# - Microphysics wrappers calling CloudMicrophysics.jl
# - Unified microphysics tendencies (cloud condensate + precipitation)
# - Moisture fixers

# Core utilities (used by other files)
include("sgs_quadrature.jl")
include("sgs_saturation.jl")
include("tendency_limiters.jl")

# Cloud diagnostics
include("cloud_fraction.jl")

# Microphysics processes
include("microphysics_wrappers.jl")
include("tendency.jl")
include("moisture_fixers.jl")
