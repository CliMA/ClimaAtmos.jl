using Test
import Adapt
import CUDA
import JET

include(joinpath(@__DIR__, "test_compilation.jl"))
using .TestCompilation
using .TestCompilation: CompilationTestFailure, Finding, IssueStageReport, JETStageReport

const FT = Float64

# Stable on every stage: no indexing (scalar indexing of host CuArrays is
# dynamic in GPUArraysCore), no throw paths with runtime-computed arguments.
check_length(a) = (length(a) == typemax(Int) && error("too long"); nothing)

# Stable everywhere except when `a` is a host CuArray (the `:host` stage), so
# it is checked with `stages` that exclude `:host`.
function stable_fill!(a, v)
    for i in eachindex(a)
        @inbounds a[i] = v
    end
    return nothing
end

# Genuinely unstable on the CPU: JET-opt flags the dynamic `+`.
unstable_getindex(r) = (r[] + 1; nothing)

# Stable over Array but not over CuArray: GPUArrays turns contiguous views of
# CuArrays into derived CuArrays, whose type is not inferrable, so calling any
# function on such a view requires runtime dispatch. The instability lives in
# CUDA.jl frames, which the `:host` stage ignores by default.
derived_view(array) = (isempty(view(array, 1:2)); nothing)

# The hidden value type makes fill! dispatch dynamically in the kernel body.
hidden_value_fill!(a, v) = (fill!(a, Base.inferencebarrier(v)); nothing)

# A struct with an array field but no adapt rule: the array survives
# adaptation, which would be an illegal host pointer inside a kernel.
struct MissingAdaptRule
    values::Vector{FT}
end

# A struct whose adapt rule hides the converted type from inference, so its
# GPU host type cannot be computed.
struct NotInferrableAdapt
    values::Vector{FT}
end
Adapt.adapt_structure(to, x::NotInferrableAdapt) =
    Base.inferencebarrier(NotInferrableAdapt(Adapt.adapt(to, x.values)))

# Host functions that launch kernels via CUDA.@cuda (only ever inferred by the
# tests, never executed, so no GPU is needed). The bad one's kernel closure
# captures a `Type`, which is not isbits, so only a real launch would crash —
# no stage other than the launch-extraction check can see it.
good_launcher!(x) = (kern = y -> nothing; CUDA.@cuda kern(x); nothing)
function bad_launcher!(x)
    T = eltype(x)
    kern = y -> (Base.donotdelete(T); nothing)
    CUDA.@cuda kern(x)
    return nothing
end
deep_launcher!(x) = (good_launcher!(x); nothing) # launch below the entry point
function unstable_kernel_launcher!(x)
    kern = y -> (@inbounds y[1] = Base.inferencebarrier(one(FT)); nothing)
    CUDA.@cuda kern(x)
    return nothing
end

do_nothing(x) = nothing

# Dummy singleton types for the type_replacements keyword.
abstract type AbstractDummyDevice end
struct DummyCPUDevice <: AbstractDummyDevice end
struct DummyGPUDevice <: AbstractDummyDevice end
struct DeviceHolder{D}
    device::D
end

host_cuarray(T, N) = CUDA.CuArray{T, N, CUDA.DeviceMemory}

# Assert that the selected stages pass, logging any reports for debugging.
function check_passes(f, args; kwargs...)
    ok, reports = compilation_reports(f, args; kwargs...)
    isempty(reports) || @info "unexpected reports" reports
    @test ok
end

# Assert that the selected stages fail; return the reports for further checks.
function failing_reports(f, args; kwargs...)
    ok, reports = compilation_reports(f, args; kwargs...)
    @test !ok
    return reports
end

@testset "TestCompilation" begin
    @testset "all stages pass for a stable call" begin
        check_passes(check_length, (zeros(FT, 4),))
        @test_compilation check_length(zeros(FT, 4))
        @test_compilation stages = (:cpu, :kernel, :pointers, :llvm_types) stable_fill!(
            zeros(FT, 4),
            one(FT),
        )
    end

    @testset "cpu stage matches JET.@test_opt" begin
        reports = failing_reports(unstable_getindex, (Ref{Any}(1),); stages = (:cpu,))
        @test all(r -> r.stage == :cpu && r isa JETStageReport, reports)
        @test !isempty(JET.get_reports(reports[1].result))
    end

    @testset "host stage catches CuArray-only instability" begin
        array = zeros(FT, 4)
        check_passes(derived_view, (array,); stages = (:cpu,), host_ignored_modules = ())
        reports = failing_reports(
            derived_view,
            (array,);
            stages = (:host,),
            host_ignored_modules = (),
        )
        @test all(r -> r.stage == :host, reports)
        # The default ignored modules hide the CUDA-internal instability.
        check_passes(derived_view, (array,); stages = (:host,))
    end

    @testset "host types are computed by Adapt inference" begin
        @test TestCompilation.host_type(Vector{FT}) == host_cuarray(FT, 1)
        @test TestCompilation.host_type(Matrix{Int}) == host_cuarray(Int, 2)
        @test TestCompilation.host_type(FT) == FT
        # Wrappers are converted through their own adapt rules.
        @test TestCompilation.host_type(Tuple{Vector{FT}, Int}) ==
              Tuple{host_cuarray(FT, 1), Int}
        # The host_array_type hook overrides the conversion target.
        @test TestCompilation.host_type(Vector{FT}; host_array_type = Array) ==
              Vector{FT}
    end

    @testset "type_replacements swaps device singletons" begin
        replacements = (AbstractDummyDevice => DummyGPUDevice,)
        @test TestCompilation.host_type(
            DummyCPUDevice;
            type_replacements = replacements,
        ) == DummyGPUDevice
        # Replacements apply structurally, inside type parameters.
        @test TestCompilation.host_type(
            DeviceHolder{DummyCPUDevice};
            type_replacements = replacements,
        ) == DeviceHolder{DummyGPUDevice}
    end

    @testset "host stage reports non-inferrable GPU types" begin
        holder = NotInferrableAdapt(zeros(FT, 2))
        report = only(failing_reports(do_nothing, (holder,); stages = (:host,)))
        @test report isa IssueStageReport && report.stage == :host
        @test occursin("not inferrable", report.findings[1].message)
        @test occursin("NotInferrableAdapt", report.findings[1].message)
    end

    @testset "kernel stage catches device-incompatible calls" begin
        args = (zeros(FT, 4), one(FT))
        reports = failing_reports(hidden_value_fill!, args; stages = (:kernel,))
        @test all(r -> r.stage == :kernel, reports)
        messages = sprint.(show, reports)
        # NOTE: `occursin(x)` curries the haystack, not the needle.
        @test any(m -> occursin("whole call as kernel body", m), messages)
        @test any(m -> occursin("dynamic", m), messages)
    end

    @testset "pointer stage catches arrays that Adapt leaves on the host" begin
        holder = MissingAdaptRule(zeros(FT, 3))
        reports = failing_reports(do_nothing, (holder,); stages = (:pointers,))
        @test reports[1] isa IssueStageReport && reports[1].stage == :pointers
        @test occursin(".values", reports[1].findings[1].message)

        # Plain arrays are converted to device arrays by the stand-in rule.
        check_passes(do_nothing, (zeros(FT, 3),); stages = (:pointers,))
    end

    @testset "kernel argument conversion" begin
        adapted = TestCompilation.kernel_arguments((zeros(FT, 2, 3),))[1]
        @test adapted isa CUDA.CuDeviceArray{FT, 2}
        @test size(adapted) == (2, 3)
    end

    @testset "kernel launch extraction" begin
        sig = Tuple{typeof(good_launcher!), host_cuarray(FT, 1)}
        sites = TestCompilation.extract_kernel_launches(sig)
        @test length(sites) == 1
        @test TestCompilation.is_resolved(sites[1])
        @test sites[1].kernel_type <: Function
        @test sites[1].arg_types == Tuple{CUDA.CuDeviceVector{FT, 1}}
        @test occursin("test_compilation_tests.jl", sites[1].location)

        # Launches are found through intermediate host functions.
        deep_sig = Tuple{typeof(deep_launcher!), host_cuarray(FT, 1)}
        deep_sites = TestCompilation.extract_kernel_launches(deep_sig)
        @test length(deep_sites) == 1
        @test deep_sites[1].kernel_type == sites[1].kernel_type

        # Functions that launch nothing have no launch sites.
        @test isempty(
            TestCompilation.extract_kernel_launches(
                Tuple{typeof(check_length), host_cuarray(FT, 1)},
            ),
        )
    end

    @testset "launch-boundary closure captures are detected" begin
        # The good twin passes: its kernel closure captures nothing. This also
        # shows that the extracted launch replaces the whole-call analysis,
        # which would reject good_launcher!'s own launch machinery.
        check_passes(good_launcher!, (zeros(FT, 4),))
        @test_compilation good_launcher!(zeros(FT, 4))

        reports = failing_reports(bad_launcher!, (zeros(FT, 4),); stages = (:kernel,))
        @test reports[1] isa IssueStageReport && reports[1].stage == :kernel
        message = reports[1].findings[1].message
        @test occursin("captures non-isbits", message)
        @test occursin("Type{$FT}", message) # the offending field type
        @test occursin("non-bitstype argument", message)
    end

    @testset "extracted kernel bodies are compiled and JET-analyzed" begin
        args = (zeros(FT, 4),)
        reports = failing_reports(unstable_kernel_launcher!, args; stages = (:kernel,))
        labels = [r.label for r in reports]
        # The failures are attributed to the extracted kernel, not the host.
        @test any(l -> occursin("kernel 1", l), labels)
        @test any(l -> occursin("IR validation", l), labels)
        @test any(l -> occursin("device method table", l), labels)
    end

    @testset "llvm_types stage" begin
        llvm_findings(T; llvm) =
            TestCompilation.llvm_type_findings!(Finding[], T, "args[1]"; llvm)

        # Int128/UInt128 in a kernel parameter: fixed in LLVM 20.
        for T in (Int128, UInt128, Tuple{UInt128, FT}, @NamedTuple{a::Int128})
            @test !isempty(llvm_findings(T; llvm = v"15"))
            @test isempty(llvm_findings(T; llvm = v"20"))
        end
        message = llvm_findings(Tuple{FT, UInt128}; llvm = v"15")[1].message
        @test occursin("args[1][2]", message)
        @test occursin("llvm/llvm-project#49221", message)

        # Non-power-of-two SIMD vectors: fixed in LLVM 19.
        V3 = NTuple{3, Core.VecElement{FT}}
        V4 = NTuple{4, Core.VecElement{FT}}
        @test !isempty(llvm_findings(V3; llvm = v"18"))
        @test occursin(
            "llvm/llvm-project#104524",
            llvm_findings(V3; llvm = v"18")[1].message,
        )
        @test isempty(llvm_findings(V3; llvm = v"19"))
        @test isempty(llvm_findings(V4; llvm = v"18"))
        @test isempty(llvm_findings(FT; llvm = v"15"))

        # The stage itself runs on the adapted kernel argument types, gated on
        # the LLVM version that ships with this Julia.
        int128_args = (fill(UInt128(0), 4), UInt128(1))
        if Base.libllvm_version < v"20"
            reports = failing_reports(stable_fill!, int128_args; stages = (:llvm_types,))
            @test reports[1] isa IssueStageReport && reports[1].stage == :llvm_types
            @test occursin("args[2]", reports[1].findings[1].message)
        else
            check_passes(stable_fill!, int128_args; stages = (:llvm_types,))
        end
        check_passes(stable_fill!, (zeros(FT, 4), one(FT)); stages = (:llvm_types,))
    end

    @testset "unknown stages are rejected" begin
        @test_throws ArgumentError compilation_reports(
            do_nothing,
            (1,);
            stages = (:ptx,),
        )
    end

    @testset "JET-style failure output" begin
        reports = failing_reports(unstable_getindex, (Ref{Any}(1),); stages = (:cpu,))
        str = sprint(show, reports[1])
        @test occursin("── [cpu]", str) # stage header
        @test occursin("runtime dispatch", str) # JET's own report printer
        color_str = sprint(show, reports[1]; context = :color => true)
        @test occursin("\e[", color_str) # color codes when supported

        # A failing @test_compilation records one organized report.
        old_print_enable = Test.TESTSET_PRINT_ENABLE[]
        Test.TESTSET_PRINT_ENABLE[] = false
        inner_testset = Test.DefaultTestSet("inner")
        Test.push_testset(inner_testset)
        local testres
        try
            testres =
                @test_compilation stages = (:cpu,) unstable_getindex(Ref{Any}(1))
        finally
            Test.pop_testset()
            Test.TESTSET_PRINT_ENABLE[] = old_print_enable
        end
        @test testres isa CompilationTestFailure
        @test length(inner_testset.results) == 1
        @test inner_testset.results[1] isa Test.Fail
        failure_str = sprint(show, testres)
        @test occursin("Compilation test failed", failure_str)
        @test occursin("@test_compilation", failure_str)
        @test occursin("unstable_getindex", failure_str)
        @test occursin("── [cpu]", failure_str)

        # A passing @test_compilation records a Pass.
        pass_testset = Test.DefaultTestSet("inner pass")
        Test.push_testset(pass_testset)
        try
            @test_compilation stages = (:cpu,) check_length(zeros(FT, 2))
        finally
            Test.pop_testset()
        end
        @test pass_testset.n_passed == 1
    end
end

# =============================================================================
# ClimaCore integration (everything above runs without ClimaCore; this section
# is skipped automatically in environments that do not provide ClimaCore)
# =============================================================================

if isnothing(Base.find_package("ClimaCore"))
    @info "Skipping ClimaCore integration tests (ClimaCore not in environment)"
else

    import ClimaCore
    import ClimaCore: DataLayouts
    import ClimaComms

    if !isdefined(DataLayouts, :DataScope)
        @info "Skipping ClimaCore integration tests (they require the unified \
               DataLayouts API, which this version of ClimaCore does not have)"
    else

        @testset "ClimaCore integration" begin
            data = DataLayouts.VIJFH{FT, 3, 4, 4, nothing}(Array{FT}, 5)

            @testset "host types through ClimaCore's adapt rules" begin
                HT = TestCompilation.host_type(typeof(data))
                @test isconcretetype(HT)
                @test HT <: DataLayouts.DataLayout
                @test HT.parameters[end] <: CUDA.CuArray{FT, 5}
                # The scope is recomputed by ClimaCore's own adapt rules.
                @test HT.parameters[end - 1] !== typeof(DataLayouts.DataScope(data))
            end

            @testset "device singleton replacement" begin
                @test TestCompilation.host_type(
                    ClimaComms.CPUSingleThreaded;
                    type_replacements = (
                        ClimaComms.AbstractCPUDevice => ClimaComms.CUDADevice,
                    ),
                ) == ClimaComms.CUDADevice
            end

            @testset "all stages pass for a DataLayout call" begin
                check_passes(fill!, (data, one(FT)))
                @test_compilation fill!(data, one(FT))
            end

            @testset "kernel stage catches device-incompatible calls" begin
                reports = failing_reports(
                    hidden_value_fill!,
                    (data, one(FT));
                    stages = (:kernel,),
                )
                @test all(r -> r.stage == :kernel, reports)
            end

            @testset "kernel argument conversion applies ClimaCore adapt rules" begin
                adapted = TestCompilation.kernel_arguments((data,))[1]
                @test parent(adapted) isa CUDA.CuDeviceArray{FT, 5}
                check_passes(do_nothing, (data,); stages = (:pointers,))
            end
        end

    end # unified DataLayouts API guard
end # ClimaCore integration guard
