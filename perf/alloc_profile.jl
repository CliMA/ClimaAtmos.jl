redirect_stderr(IOContext(stderr, :stacktrace_types_limited => Ref(false)))
import Random
import ClimaComms
ClimaComms.@import_required_backends
Random.seed!(1234)
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP

include(joinpath(@__DIR__, "..", "test", "test_helpers.jl"))

println("=== 2M Microphysics Allocation Test ===\n")

config = CA.AtmosConfig(
    Dict(
        "initial_condition" => "PrecipitatingColumn",
        "moist" => "nonequil",
        "precip_model" => "2M",
        "config" => "column",
        "output_default_diagnostics" => false,
        "prescribed_aerosols" => ["SSLT01"],
    ),
    job_id = "alloc_2m",
)
(; Y, p) = generate_test_simulation(config)
FT = eltype(Y)
t = FT(0)

# Setup
CA.set_precipitation_velocities!(Y, p, p.atmos.moisture_model, p.atmos.microphysics_model, p.atmos.turbconv_model)
CA.set_microphysics_tendency_cache!(Y, p, p.atmos.microphysics_model, p.atmos.turbconv_model)

# Warmup set_precomputed_quantities!
for _ in 1:5; CA.set_precomputed_quantities!(Y, p, t); end
a = [(@allocated CA.set_precomputed_quantities!(Y, p, t)) for _ in 1:10]
println("set_precomputed_quantities! (2M): min=$(minimum(a)) bytes")

# Warmup microphysics_tendency!
ᶜYₜ = zero(Y)
(; turbconv_model, moisture_model, microphysics_model) = p.atmos
for _ in 1:3
    ᶜYₜ .= zero(eltype(ᶜYₜ))
    CA.microphysics_tendency!(ᶜYₜ, Y, p, FT(0), moisture_model, microphysics_model, turbconv_model)
end
am = [begin
    ᶜYₜ .= zero(eltype(ᶜYₜ))
    @allocated CA.microphysics_tendency!(ᶜYₜ, Y, p, FT(0), moisture_model, microphysics_model, turbconv_model)
end for _ in 1:10]
println("microphysics_tendency! (2M): min=$(minimum(am)) bytes")

# Warmup set_microphysics_tendency_cache!
for _ in 1:3
    CA.set_microphysics_tendency_cache!(Y, p, microphysics_model, turbconv_model)
end
ac = [(@allocated CA.set_microphysics_tendency_cache!(Y, p, microphysics_model, turbconv_model)) for _ in 1:10]
println("set_microphysics_tendency_cache! (2M): min=$(minimum(ac)) bytes")

# ============================
# 0M benchmark for regression
# ============================
println("\n=== 0M Benchmark ===\n")
config0 = CA.AtmosConfig(
    CA.YAML.load_file("config/perf_configs/bm_default.yml");
    job_id = "alloc_0m",
)
(; Y, p) = generate_test_simulation(config0)
FT0 = eltype(Y)
t0 = FT0(0)

for _ in 1:5; CA.set_precomputed_quantities!(Y, p, t0); end
a0 = [(@allocated CA.set_precomputed_quantities!(Y, p, t0)) for _ in 1:10]
println("set_precomputed_quantities! (0M): min=$(minimum(a0)) bytes")
