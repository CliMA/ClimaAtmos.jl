# offline_bins.jl -- drive a many-bin sea-salt column OFFLINE from the saved
# state of an online ClimaAtmos column run.
#
# usage: julia --project=<env> offline_bins.jl <offline_config.yml> \
#            <online_state_dir> <out.nc> [output_every_s=600]
#
# The offline config builds a column model with an EXPLICIT ode_algo (so no
# Jacobian is compiled, which is what makes 50 bins affordable) and `dt` equal
# to the aerosol SUB-step. The CTS solver is never called. Instead, for every
# consecutive pair of saved online states (dayD.S.hdf5, written every online
# step), the non-aerosol prognostic fields are linearly interpolated in time
# and copied into the offline state, the cache is rebuilt with
# set_precomputed_quantities!, and the full model tendency
# (implicit_tendency! + remaining_tendency!) is evaluated; only the aerosol
# components of that tendency are applied, with Heun's method (SSP RK2) on the
# sub-step, followed by the model's own non-negativity clamp. The aerosols are
# passive, so freezing the dynamics within a step is the only approximation
# (first-order splitting, reduced by the interpolation).
#
# Output: a NetCDF file with (time, z) grid-mean mass mixing ratios of every
# bin and history tracer, the updraft specific values, and a few dynamics
# fields for reference.

import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaCore: Fields, InputOutput
import NCDatasets
import Logging
import Dates

config_file, state_dir, out_file = ARGS[1], ARGS[2], ARGS[3]
output_every = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 600.0

t0_wall = time()
elapsed() = round(time() - t0_wall; digits = 1)
say(msg) = (println("[t+$(elapsed())s] ", msg); flush(stdout))

# ---------------------------------------------------------------------------
# Build the offline model (explicit algorithm => no Jacobian).
# ---------------------------------------------------------------------------
job_id = "offline_" * replace(splitext(basename(out_file))[1], r"[^A-Za-z0-9_]" => "_")
config = CA.AtmosConfig(config_file; job_id)
sim = CA.AtmosSimulation(config)
say("built offline model (PROBE_BUILD_SECONDS=$(elapsed()))")
(; integrator) = sim
Y = integrator.u
p = integrator.p
FT = eltype(Y)
dt_sub = float(p.dt)
ctx = ClimaComms.context(Y.c)

# Aerosol names: grid-mean `ρ<bin>` symbols and updraft `<bin>` symbols.
species = filter(m -> m isa CA.AbstractPrognosticAerosol, collect(values(CA.species_models(p.atmos.aerosols))))
bin_syms = Tuple(Symbol(n) for m in species for n in CA.bin_names(m))
gs_aero = Set(Symbol(:ρ, n) for n in bin_syms)
sgs_aero = Set(bin_syms)
say("aerosol tracers: $(length(bin_syms)) ($(first(bin_syms)) .. $(last(bin_syms))), dt_sub = $dt_sub s")

# ---------------------------------------------------------------------------
# Saved online states.
# ---------------------------------------------------------------------------
files = filter(f -> occursin(r"^day\d+\.\d+\.hdf5$", basename(f)), readdir(state_dir; join = true))
isempty(files) && error("no dayD.S.hdf5 files in $state_dir")
times = CA.time_from_filename.(files)
order = sortperm(times)
files, times = files[order], times[order]
say("$(length(files)) online states, t = $(first(times)) .. $(last(times)) s")

function read_state(file)
    reader = InputOutput.HDF5Reader(file, ctx)
    Yon = InputOutput.read_field(reader, "Y")
    t = InputOutput.HDF5.read_attribute(reader.file, "time")
    Base.close(reader)
    return Yon, Float64(t)
end

# ---------------------------------------------------------------------------
# Field plumbing. Everything goes through `parent` arrays so that fields read
# from HDF5 (their own space object) and fields of the offline model can be
# mixed without triggering ClimaCore's space-identity check.
# ---------------------------------------------------------------------------
sgs1(x) = getproperty(x.sgsʲs, 1)   # first updraft of a center/face FieldVector part

"""Blend the non-aerosol prognostic fields: Y ← (1-α)·Ya + α·Yb."""
function blend_dynamics!(Y, Ya, Yb, α)
    for n in propertynames(Ya.c)
        n === :sgsʲs && continue
        n in gs_aero && continue
        d, a, b = parent(getproperty(Y.c, n)), parent(getproperty(Ya.c, n)), parent(getproperty(Yb.c, n))
        @. d = (1 - α) * a + α * b
    end
    for n in propertynames(sgs1(Ya.c))
        n in sgs_aero && continue
        d, a, b = parent(getproperty(sgs1(Y.c), n)), parent(getproperty(sgs1(Ya.c), n)), parent(getproperty(sgs1(Yb.c), n))
        @. d = (1 - α) * a + α * b
    end
    for n in propertynames(Ya.f)
        if n === :sgsʲs
            d, a, b = parent(sgs1(Y.f).u₃), parent(sgs1(Ya.f).u₃), parent(sgs1(Yb.f).u₃)
        else
            d, a, b = parent(getproperty(Y.f, n)), parent(getproperty(Ya.f, n)), parent(getproperty(Yb.f, n))
        end
        @. d = (1 - α) * a + α * b
    end
    return nothing
end

aero_gs_fields(Y) = [parent(getproperty(Y.c, n)) for n in bin_syms_ρ]
const bin_syms_ρ = Tuple(Symbol(:ρ, n) for n in bin_syms)
aero_sgs_fields(Y) = [parent(getproperty(sgs1(Y.c), n)) for n in bin_syms]
aero_arrays(Y) = vcat(aero_gs_fields(Y), aero_sgs_fields(Y))

Yₜ_imp = similar(Y)
Yₜ_exp = similar(Y)
Yₜ_lim = similar(Y)
"""Full model tendency at the current Y; returns the aerosol parts of Yₜ_imp + Yₜ_exp + Yₜ_lim."""
function aerosol_tendency!(k, Y, p, t)
    CA.set_precomputed_quantities!(Y, p, t)
    CA.implicit_tendency!(Yₜ_imp, Y, p, t)
    CA.remaining_tendency!(Yₜ_exp, Yₜ_lim, Y, p, t)
    for (kk, ai, ae, al) in zip(k, aero_arrays(Yₜ_imp), aero_arrays(Yₜ_exp), aero_arrays(Yₜ_lim))
        @. kk = ai + ae + al
    end
    return nothing
end
"""The online run's per-step aerosol constraints: the grid-mean non-negativity
clamp, and the updraft filter of `enforce_edmf_updraft_constraints!` (relax to
the grid mean where ρa is negligible, floor at zero, bound ρaχʲ ≤ ρχ)."""
function clamp_aerosols!(Y)
    CA.aerosol_nonnegativity_constraint!(Y, p, FT(0))
    ϵ = CA.ϵ_numerics(FT)
    ρ, ρa = Y.c.ρ, sgs1(Y.c).ρa
    for n in bin_syms
        χʲ = getproperty(sgs1(Y.c), n)
        ρχ = getproperty(Y.c, Symbol(:ρ, n))
        @. χʲ = ifelse(ρa < ϵ, ρχ / ρ, min(max(0, χʲ), max(0, ρχ) / ρa))
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Output buffers.
# ---------------------------------------------------------------------------
z_c = vec(parent(Fields.coordinate_field(Y.c).z))
nz = length(z_c)
out_t = Float64[]
out_gs = Dict(n => Vector{Vector{FT}}() for n in bin_syms)      # mmr = ρχ/ρ
out_sgs = Dict(n => Vector{Vector{FT}}() for n in bin_syms)     # updraft specific value
dyn_names = (:ρ, :ρtke, :ρq_tot, :ρq_lcl, :ρq_rai)
out_dyn = Dict(n => Vector{Vector{FT}}() for n in dyn_names)
out_ρa = Vector{Vector{FT}}()
function record!(t)
    push!(out_t, t)
    ρ = vec(parent(Y.c.ρ))
    for n in bin_syms
        push!(out_gs[n], vec(parent(getproperty(Y.c, Symbol(:ρ, n)))) ./ ρ)
        push!(out_sgs[n], vec(parent(getproperty(sgs1(Y.c), n))))
    end
    for n in dyn_names
        push!(out_dyn[n], vec(parent(getproperty(Y.c, n))))
    end
    push!(out_ρa, vec(parent(sgs1(Y.c).ρa)))
    return nothing
end

# ---------------------------------------------------------------------------
# Time loop.
# ---------------------------------------------------------------------------
Y_prev = copy(Y)               # the online run starts from the same initial condition
t_prev = 0.0
record!(0.0)
next_out = output_every
aero = aero_arrays(Y)
aero0 = [copy(a) for a in aero]
k1 = [similar(a) for a in aero]
k2 = [similar(a) for a in aero]
nfiles = length(files)
n_sub_total = 0
for (i, file) in enumerate(files)
    global Y_prev, t_prev, next_out, n_sub_total
    Y_next, t_next = read_state(file)
    nsub = max(1, round(Int, (t_next - t_prev) / dt_sub))
    h = (t_next - t_prev) / nsub
    for s in 1:nsub
        t_a = t_prev + (s - 1) * h
        # Heun: predictor with the dynamics at the start of the sub-step,
        # corrector with the dynamics at its end.
        for (a0, a) in zip(aero0, aero); copyto!(a0, a); end
        blend_dynamics!(Y, Y_prev, Y_next, (s - 1) / nsub)
        aerosol_tendency!(k1, Y, p, t_a)
        for (a, a0, kk) in zip(aero, aero0, k1); @. a = a0 + h * kk; end
        clamp_aerosols!(Y)
        blend_dynamics!(Y, Y_prev, Y_next, s / nsub)
        aerosol_tendency!(k2, Y, p, t_a + h)
        for (a, a0, kk1, kk2) in zip(aero, aero0, k1, k2); @. a = a0 + (h / 2) * (kk1 + kk2); end
        clamp_aerosols!(Y)
        n_sub_total += 1
    end
    Y_prev, t_prev = Y_next, t_next
    if t_next + 1e-6 >= next_out
        record!(t_next)
        next_out += output_every
    end
    if i % 100 == 0 || i == nfiles
        say("step $i/$nfiles  t = $(t_next) s  ($n_sub_total sub-steps)")
    end
end
say("integration done (PROBE_RUN_SECONDS=$(elapsed()))")

# ---------------------------------------------------------------------------
# Write NetCDF.
# ---------------------------------------------------------------------------
NCDatasets.NCDataset(out_file, "c") do ds
    NCDatasets.defDim(ds, "time", length(out_t))
    NCDatasets.defDim(ds, "z", nz)
    tv = NCDatasets.defVar(ds, "time", Float64, ("time",)); tv[:] = out_t
    zv = NCDatasets.defVar(ds, "z", Float64, ("z",)); zv[:] = z_c
    stack(v) = reduce(hcat, v; init = zeros(FT, nz, 0))   # (z, time)
    for n in bin_syms
        v = NCDatasets.defVar(ds, "mmr$(n)", FT, ("z", "time")); v[:, :] = stack(out_gs[n])
        v = NCDatasets.defVar(ds, "up$(n)", FT, ("z", "time")); v[:, :] = stack(out_sgs[n])
    end
    dyn_out_names = (; ρ = "rho", ρtke = "rho_tke", ρq_tot = "rho_q_tot", ρq_lcl = "rho_q_lcl", ρq_rai = "rho_q_rai")
    for n in dyn_names
        v = NCDatasets.defVar(ds, dyn_out_names[n], FT, ("z", "time")); v[:, :] = stack(out_dyn[n])
    end
    v = NCDatasets.defVar(ds, "rhoa_up", FT, ("z", "time")); v[:, :] = stack(out_ρa)
    ds.attrib["dt_sub"] = dt_sub
    ds.attrib["online_state_dir"] = state_dir
    ds.attrib["offline_config"] = config_file
    ds.attrib["n_bins"] = length(bin_syms)
    ds.attrib["created"] = string(Dates.now())
end
say("wrote $out_file")
println("PROBE_RET=success")
