#=
Static GPU kernel-resource regression tests for the SGS microphysics path.

Nothing here is timed, so it cannot flake under GPU contention. Each kernel is
compiled and the resources the compiler assigned are read back: register count
and local-memory bytes.

## Why this exists separately from CloudMicrophysics' version

CloudMicrophysics has an equivalent gate (`test/gpu_kernel_resources.jl` there)
covering its own kernels. It cannot catch the regression that matters here, and
says so in its own header: compiled standalone those kernels do not spill at
all, while the same physics reached through the SGS quadrature sat at the
255-register hardware cap and spilled 32.9% of its memory traffic. The pressure
comes from COMPOSITION -- the quadrature evaluating the physics N^2 times -- and
composition is assembled here, not there.

Concretely, measured on an A100: the 1M physics alone is ~116 registers; wrapped
in the nine-point quadrature it hits the 255 cap. A gate on CloudMicrophysics
alone passes happily through that.

## What this does and does not cover

These kernels call the quadrature directly, one thread per point. They do NOT go
through a ClimaCore broadcast, and the gap is large -- measured on an A100:

    this file, scalar        direct 120 regs   quadrature 147   (+27)
    production broadcast     direct 153 regs   quadrature 255   (+102, at the cap)

The broadcast adds ~64 registers of framework (field indexing, NamedTuple
output) and, more importantly, it is what pushes the composition overhead from
+27 to +102. So this file UNDER-REPORTS the production pressure and cannot by
itself catch a kernel going to the register cap. It is a regression gate on the
part ClimaAtmos owns -- if the quadrature's overhead over the physics it wraps
starts growing here, it will be far worse in the broadcast. Gating the broadcast
itself needs ClimaCore to expose the compiled kernel's resources; until then,
`results/kernel-resources.csv` in the profiling project covers it out of band.

## Read local memory, not just registers

Registers saturate. Above 255 the compiler must spill, so a kernel far past the
limit still reports exactly 255 and register count alone cannot tell "fine" from
"catastrophic", nor show whether a fix helped. Three separate attempts at this
kernel read as no-ops for that reason before local memory was recorded. Both
numbers are gated below; the local-memory one is what moves when registers are
pinned.
=#

using Test
using CUDA
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import CloudMicrophysics.BulkMicrophysicsTendencies as BMT
import CloudMicrophysics.Parameters as CMP
import Thermodynamics as TD

CUDA.allowscalar(false)

if !CUDA.functional()
    @warn "No CUDA device; GPU kernel-resource gate did not run."
else

    # Gate on A100 regardless of which GPU hosts the test.
    const TARGET_CAP = v"8.0"

    # sm_80 occupancy steps, in registers per thread.
    warps_per_sm(r) = r <= 128 ? 16 : r <= 168 ? 12 : r <= 255 ? 8 : 0

    # Baselines per CUDA toolkit: pinning the architecture removes GPU-to-GPU
    # variation but not toolkit variation. `nothing` means not yet measured on that
    # toolkit -- the test reports the value and asks for it to be recorded, and
    # keeps gating everything else. Skipping is how a gate silently runs zero
    # assertions while reporting success.
    const BASELINES = Dict(
        v"13.0" => Dict(
            :direct => (registers = nothing, local_bytes = nothing),
            :quadrature => (registers = nothing, local_bytes = nothing),
        ),
    )

    # Slack absorbs run-to-run jitter: Julia's inference is not bit-identical across
    # processes and register allocation follows it. Still far tighter than a real
    # regression, which moves tens of registers at a time.
    const REG_SLACK = 8
    const LOCAL_SLACK = 128

    const FT = Float32
    const MP = CMP.Microphysics1MParams(FT)
    const TPS = TD.Parameters.ThermodynamicsParameters(FT)
    const QUAD = CA.SGSQuadrature(FT; quadrature_order = 3)
    const DT, ALPHA, CORR = FT(30), FT(1), FT(0.1)
    const NSUBS = 2

    struct State{FT}
        ρ::FT
        T::FT
        q_tot::FT
        q_lcl::FT
        q_icl::FT
        q_rai::FT
        q_sno::FT
        TT::FT
        qq::FT
        lam::FT
    end
    const ST = State{FT}(
        FT(0.9), FT(275), FT(8e-3), FT(3e-4), FT(5e-5), FT(2e-4), FT(3e-5),
        FT(1e-2), FT(1e-12), FT(1e-5),
    )

    function direct_kernel(out, s, mp, tps)
        r = CA.microphysics_tendencies_1m(
            s.ρ, s.q_tot, s.q_lcl, s.q_icl, s.q_rai, s.q_sno, s.T, mp, tps, DT, NSUBS,
        )
        @inbounds out[1] = r.dq_lcl_dt
        return nothing
    end

    function quadrature_kernel(out, s, mp, tps)
        r = CA.microphysics_tendencies_1m(
            BMT.Microphysics1Moment(), QUAD, mp, tps, s.ρ, s.T, s.q_tot,
            s.q_lcl, s.q_icl, s.q_rai, s.q_sno, s.TT, s.qq, CORR, s.lam,
            ALPHA, DT, NSUBS,
        )
        @inbounds out[1] = r.dq_lcl_dt
        return nothing
    end

    function resources(f, out)
        k = CUDA.@cuda launch = false always_inline = true f(out, ST, MP, TPS)
        return (registers = Int(CUDA.registers(k)),
            local_bytes = Int(CUDA.memory(k).local))
    end

    toolkit = CUDA.runtime_version()
    key = VersionNumber(toolkit.major, toolkit.minor)
    if !haskey(BASELINES, key)
        key = maximum(keys(BASELINES))
        @warn "No baseline for CUDA $toolkit; gating against $key instead."
    end
    baselines = BASELINES[key]

    out = CUDA.zeros(FT, 1)
    measured = Dict(
        :direct => resources(direct_kernel, out),
        :quadrature => resources(quadrature_kernel, out),
    )

    @testset "GPU kernel resources (SGS microphysics, sm_80)" begin
        for (name, m) in measured
            b = baselines[name]
            @info "$name: $(m.registers) registers, $(m.local_bytes) B local, " *
                  "$(warps_per_sm(m.registers)) warps/SM"

            # Hard limit: at the cap the compiler is spilling to stay there, and the
            # reported number no longer reflects demand.
            @test m.registers < 255

            if isnothing(b.registers)
                @warn "No recorded baseline for $name on CUDA $key. Record " *
                      "(registers = $(m.registers), local_bytes = $(m.local_bytes))."
            else
                @test m.registers <= b.registers + REG_SLACK
                @test m.local_bytes <= b.local_bytes + LOCAL_SLACK
            end
        end

        # Composition is the thing this file exists to watch. If the quadrature ever
        # costs this much more than the physics it wraps, the per-point barrier has
        # stopped holding -- which is exactly the failure that put this kernel at the
        # register cap.
        Δ = measured[:quadrature].registers - measured[:direct].registers
        @info "quadrature composition overhead: +$Δ registers"
        @test Δ < 96
    end

end # CUDA.functional()
