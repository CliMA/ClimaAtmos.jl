"""
    TestCompilation

Device-free compilation checking for CPU and GPU code paths. No GPU is needed
for any check in this module (`CUDA.functional()` may be `false`), so tests
built on it can guarantee GPU compilation without requesting CUDA devices. The
module is package-independent: it only depends on `Adapt`, `CUDA`, `JET`, and
`Test`, and every check works on plain Julia functions and `Array`s.

Given a call `f(args...)` with CPU (`Array`-backed) arguments, this module can
run five analyses:

 1. `:cpu` — JET's optimization analysis of the call itself, equivalent to
    `JET.@test_opt f(args...)`: reports every runtime dispatch or optimization
    failure over the CPU argument types.
 2. `:host` — the same JET analysis over the argument types as they would
    appear on a machine with a GPU. The GPU types are obtained by *inferring*
    (never executing, so no GPU memory is allocated)
    `Adapt.adapt(host_array_type, arg)` for each argument, which applies every
    package's own `Adapt` rules; `Array`s become `CuArray`s and wrapper
    structures are rebuilt around them. Types with no `Adapt`-visible GPU form
    (e.g. device singletons) can be swapped with the `type_replacements`
    keyword. This is what `JET.@test_opt` sees in GPU CI jobs, and it catches
    host-side instabilities in kernel launch code.
 3. `:kernel` — GPU device code analysis, in one of two modes:
      - *Launch extraction* (used when it applies): the optimized IR of
        `f(args...)` over the `:host` argument types is searched for
        `CUDA.cufunction` call sites — the lowered form of `CUDA.@cuda` — by
        recursively walking `Base.code_typed` results (bounded depth and
        breadth). Each resolved launch yields the exact kernel function type
        and device argument types, including closures constructed inside host
        functions. Each launch is checked for non-isbits kernel arguments
        (e.g. a closure capturing a `Type`, which only fails at a real
        launch), then compiled and analyzed as described below.
      - *Whole-call fallback* (when no launch is resolved from the host IR):
        the arguments are converted with the same `Adapt`/`CUDA.KernelAdaptor`
        rules that real kernel launches use (with `Array` leaves standing in
        for `CuArray`s), and the call itself is treated as a kernel body.
        This suits functions that dispatch to a device implementation when
        called on device-converted arguments (e.g. via a scope or device
        argument), which is also why launch wrappers whose `cufunction` call
        is dynamic (so the kernel closure type cannot be extracted) fall back
        to this mode instead of failing: the device code is still covered,
        though non-isbits closure captures inside such wrappers cannot be
        detected.
    In both modes, two analyses run on each kernel signature:
      - GPUCompiler's LLVM IR validation, which catches `InvalidIRError`s
        (dynamic dispatch, GPU-illegal operations) at the stage right before
        the IR would be compiled to PTX; and
      - JET's optimization analysis over CUDA's device method table, so that
        device intrinsics like `threadIdx` are not treated as dead code.
 4. `:pointers` — a scan of the adapted arguments for host arrays that
    survived adaptation. A field that remains an `Array` after the
    `KernelAdaptor` runs corresponds to a host pointer inside a kernel
    argument on a real GPU, which causes an illegal memory access at runtime
    (this cannot be caught by compilation alone).
 5. `:llvm_types` — a scan of the kernel argument types (both the adapted
    whole-call arguments and any extracted launch signatures) for Julia types
    whose LLVM lowering is known to crash the NVPTX backend in the LLVM
    version Julia ships (`Base.libllvm_version`). Compilation to LLVM IR
    succeeds for these types, so `:kernel` cannot catch them; on a real GPU
    machine they segfault LLVM during instruction selection. See
    [`llvm_type_findings!`](@ref) for the rules and their version gates.

# Usage

    using .TestCompilation

    # In a @testset (with args constructed on the CPU):
    @test_compilation fill!(data, value)
    @test_compilation stages = (:cpu, :host) my_solver_step!(state, cache)

    # Programmatically:
    ok, reports = compilation_reports(fill!, (data, value))
    ok, reports = compilation_reports(f, args; stages = (:cpu,))

Each entry of `reports` is a [`StageReport`](@ref) whose `show` method prints
a stage header followed by the underlying findings — JET results are printed
with JET's own report printer. On failure, [`@test_compilation`](@ref) records
a single organized report listing each stage's findings, in the same style as
`JET.@test_opt`.

# Keyword arguments

  - `stages = (:cpu, :host, :kernel, :pointers, :llvm_types)`: which analyses
    to run.
  - `host_array_type = CUDA.CuArray`: the array type `Array`s are converted to
    (via `Adapt`) for the `:host` stage and launch extraction.
  - `type_replacements = ()`: a tuple of `Pair`s of types applied structurally
    to the argument types before the `Adapt`-based `:host` conversion; any
    type parameter `P` with `P <: first(pair)` is replaced by `last(pair)`.
    Use this for GPU-machine differences `Adapt` cannot express, e.g.
    `(ClimaComms.AbstractCPUDevice => ClimaComms.CUDADevice,)`.
  - `host_ignored_modules = default_host_ignored_modules()`: JET
    `ignored_modules` for the host-side stages (`:cpu` and `:host`).
  - `kernel_ignored_modules = ()`: JET `ignored_modules` for the `:kernel`
    stage. Empty by default because device-side code lives in CUDA.jl and must
    be analyzed, unlike CUDA.jl's intentionally-dynamic host machinery.
  - `extra_ignored_modules = ()`: appended to both of the above; the simplest
    way for packages to hide known-dirty frames of their own.
  - `max_extraction_depth = 10`, `max_extraction_visits = 1000`: bounds on the
    recursive IR walk used by launch extraction.

All other keyword arguments are forwarded to `JET.report_opt` (e.g.
`function_filter`).
"""
module TestCompilation

import Adapt
import CUDA
import JET
import Test

const CC = Core.Compiler
# NOTE: not named `GC`: `CUDA.@cuda` expands to an unhygienic `GC.@preserve`,
# so code that uses both this module and `CUDA.@cuda` must not shadow
# `Base.GC` with `GPUCompiler`.
const GPUC = CUDA.GPUCompiler

const DEFAULT_STAGES = (:cpu, :host, :kernel, :pointers, :llvm_types)
const DEFAULT_STAGES_DOC = "(:cpu, :host, :kernel, :pointers, :llvm_types)"

export compilation_reports, @test_compilation

# ─── Findings and stage reports ──────────────────────────────────────────────

"""
    Finding(message, [location])

A single non-JET issue: a human-readable (possibly multi-line) `message` and
an optional `"file:line"` `location`.
"""
struct Finding
    message::String
    location::Union{Nothing, String}
end
Finding(message::AbstractString) = Finding(String(message), nothing)

"""
    StageReport

One failing analysis from [`compilation_reports`](@ref). Concrete subtypes are
[`JETStageReport`](@ref) (wraps a `JET.JETCallResult` and prints it with JET's
own report printer) and [`IssueStageReport`](@ref) (a list of
[`Finding`](@ref)s). Both print as a stage header followed by the findings,
with colors when the output supports them.
"""
abstract type StageReport end

"A failing JET analysis; `result` is shown with JET's organized report printer."
struct JETStageReport <: StageReport
    stage::Symbol
    label::String
    result::JET.JETCallResult
end

"Failing non-JET checks (IR validation, pointer scan, LLVM type rules, ...)."
struct IssueStageReport <: StageReport
    stage::Symbol
    label::String
    findings::Vector{Finding}
end

function print_stage_header(io::IO, report::StageReport)
    printstyled(io, "── [", report.stage, "] "; bold = true, color = :cyan)
    printstyled(io, report.label; color = :cyan)
    printstyled(io, " ──"; bold = true, color = :cyan)
    println(io)
end

indent_lines(str, indent) =
    string(indent, replace(rstrip(str, '\n'), '\n' => string('\n', indent)))

function Base.show(io::IO, report::JETStageReport)
    print_stage_header(io, report)
    # `repr` with `context = io` preserves color support, like JET's own
    # `Test` integration does.
    println(io, indent_lines(repr(report.result; context = io), "  "))
end

function Base.show(io::IO, report::IssueStageReport)
    print_stage_header(io, report)
    for finding in report.findings
        # The bullet stands in for the first line's indent.
        message = chopprefix(indent_lines(finding.message, "    "), "    ")
        print(io, "  • ", message)
        isnothing(finding.location) ||
            printstyled(io, " @ ", finding.location; color = :light_black)
        println(io)
    end
end

# ─── Default JET frame filters ───────────────────────────────────────────────

"""
    default_host_ignored_modules()

JET `ignored_modules` for the host-side stages (`:cpu` and `:host`):

  - `CUDA`: host-side kernel launch machinery (`cufunction` compilation
    caching, launch configuration, argument conversion) is intentionally
    dynamic and unavoidably reached by any code that launches kernels.
  - `GPUCompiler`: the compilation cache lookup on the launch path is dynamic
    for the same reason.
  - `Adapt`: `cudaconvert`'s structural conversion of launch arguments runs
    through `@nospecialize`d generic fallbacks on the host.
  - `Base.Threads`: task spawning passes through dynamic error paths in
    `Base.Threads`' internals. Functions parallelized over threads are still
    fully analyzed, since inference also follows the branch that runs them
    without spawning tasks.

The `:kernel` stage deliberately ignores nothing by default: device-side code
(e.g. `CuDeviceArray` indexing) lives in CUDA.jl, so ignoring CUDA frames
there would hide genuine kernel problems.
"""
default_host_ignored_modules() = (
    JET.AnyFrameModule(CUDA),
    JET.AnyFrameModule(GPUC),
    JET.AnyFrameModule(Adapt),
    JET.AnyFrameModule(Base.Threads),
)

# ─── Host-side GPU types (for the :host stage and launch extraction) ─────────

# Structurally replace type parameters: any parameter `P` with `P <: from` for
# some pair `from => to` becomes `to`. Used for GPU-machine differences that
# Adapt rules cannot express, like CPU device singletons.
function replace_types(@nospecialize(T), replacements::Tuple)
    isempty(replacements) && return T
    T isa Type || return T # non-type parameters (integers, symbols, ...)
    for (from, to) in replacements
        T <: from && return to
    end
    (T isa DataType && !isempty(T.parameters)) || return T
    params = map(P -> replace_types(P, replacements), (T.parameters...,))
    return try
        T.name.wrapper{params...}
    catch # inner constructors or parameter constraints can reject the rewrite
        T
    end
end

"""
    host_type(::Type{T}; host_array_type = CUDA.CuArray, type_replacements = ())

The type a value of type `T` would have on a machine with a GPU: the inferred
return type of `Adapt.adapt(host_array_type, x)` for `x::T` (after applying
`type_replacements`; see [`compilation_reports`](@ref)). Inference alone
computes the converted type, so no GPU memory is ever allocated. The result is
only useful when it is concrete; a non-concrete result means some `Adapt` rule
along the way is not inferrable (which [`compilation_reports`](@ref) reports
as a `:host` finding).
"""
function host_type(
    @nospecialize(T::Type);
    host_array_type::Type = CUDA.CuArray,
    type_replacements::Tuple = (),
)
    T = replace_types(T, type_replacements)
    return CC.return_type(Adapt.adapt, Tuple{Type{host_array_type}, T})
end

# The full call signature over host GPU types, or `nothing` plus findings for
# every argument (including the function object) whose GPU type cannot be
# inferred.
function host_signature(f, args; host_array_type, type_replacements)
    findings = Finding[]
    types = map(("f", map(string, 1:length(args))...), (f, args...)) do name, x
        T = host_type(typeof(x); host_array_type, type_replacements)
        Base.isdispatchtuple(Tuple{T}) && return T
        what = name == "f" ? "the function object" : "argument $name"
        push!(
            findings,
            Finding(
                "the GPU host type of $what (::$(typeof(x))) is not \
                 inferrable: `Adapt.adapt($host_array_type, ...)` infers as \
                 `$T`. Make the `Adapt` rules for this type inferrable, or \
                 map the type directly with the `type_replacements` keyword.",
            ),
        )
        return nothing
    end
    return isempty(findings) ? Tuple{types...} : nothing, findings
end

# ─── Kernel-side argument values (for :kernel, :pointers, and :llvm_types) ───

# Stand-in for CuArray -> CuDeviceArray on the real launch path: a null-pointer
# CuDeviceArray is a plain isbits struct, so it can be constructed without a
# GPU, and going through the genuine KernelAdaptor applies every
# package-specific adapt rule.
struct KernelArrayStandIn end
Adapt.adapt_storage(::KernelArrayStandIn, a::Array{T, N}) where {T, N} =
    CUDA.CuDeviceArray{T, N, CUDA.AS.Global}(
        reinterpret(Core.LLVMPtr{T, CUDA.AS.Global}, C_NULL),
        size(a),
    )
Adapt.adapt_storage(::KernelArrayStandIn, x) =
    Adapt.adapt_storage(CUDA.KernelAdaptor(), x)

# Two passes: the first converts Array leaves through the stand-in (structure
# rules specific to `::CUDA.KernelAdaptor` do not fire, since `to` is the
# stand-in); the second applies those KernelAdaptor-specific structure rules
# (Array leaves are already device arrays by then, so none remain to allocate).
kernel_arguments(args) =
    map(args) do arg
        standin = Adapt.adapt(KernelArrayStandIn(), arg)
        Adapt.adapt(CUDA.KernelAdaptor(), standin)
    end

# ─── JET over the CUDA device method table (for kernel-side analysis) ────────

# JET's default OptAnalyzer infers with the native method table, where CUDA
# intrinsics like threadIdx resolve to host definitions that just throw, so
# every kernel body looks like dead code. This report pass behaves identically
# to JET's OptAnalysisPass but routes inference through CUDA's device method
# table (and gets its own analysis cache, since JET caches per report pass).
struct DeviceOptPass <: JET.ReportPass end
(::DeviceOptPass)(T::Type{<:JET.InferenceErrorReport}, args...) =
    JET.OptAnalysisPass()(T, args...)
method_table_for(::JET.OptAnalysisPass, world::UInt) =
    CC.InternalMethodTable(world)
method_table_for(::DeviceOptPass, world::UInt) =
    GPUC.get_method_table_view(world, CUDA.method_table)
CC.method_table(analyzer::JET.OptAnalyzer) =
    method_table_for(JET.ReportPass(analyzer), JET.get_inference_world(analyzer))

function add_jet_report!(reports, stage, label, sig; device = false, jetconfigs...)
    result =
        device ?
        JET.report_opt(sig; report_pass = DeviceOptPass(), jetconfigs...) :
        JET.report_opt(sig; jetconfigs...)
    isempty(JET.get_reports(result)) ||
        push!(reports, JETStageReport(stage, label, result))
    return reports
end

# ─── GPUCompiler IR validation (for kernel-side analysis) ────────────────────

const PTX_CAP = v"7.0"
const PTX_ISA = v"7.8"

# Missing symbols that only resolve during a real kernel launch: libdevice
# math functions, GPU runtime helpers, and the kernel state intrinsic.
is_benign_ir_error(e) =
    e[1] == GPUC.UNKNOWN_FUNCTION &&
    e[3] isa AbstractString &&
    (
        startswith(e[3], "__nv") ||
        startswith(e[3], "gpu_") ||
        occursin("state_getter", e[3])
    )

# The "file:line" of the first backtrace frame with a real location.
function ir_error_location(backtrace)
    i = findfirst(frame -> frame.line > 0, backtrace)
    return isnothing(i) ? nothing : string(backtrace[i].file, ':', backtrace[i].line)
end

function ir_findings(sig::Type{<:Tuple})
    config = GPUC.CompilerConfig(
        GPUC.PTXCompilerTarget(; cap = PTX_CAP, ptx = PTX_ISA),
        CUDA.CUDACompilerParams(; cap = PTX_CAP, ptx = PTX_ISA);
        kernel = true,
        libraries = false,
        always_inline = true,
    )
    (F, arg_types...) = sig.parameters
    try
        job = GPUC.CompilerJob(GPUC.methodinstance(F, Tuple{arg_types...}), config)
        GPUC.JuliaContext() do _
            GPUC.compile(:llvm, job)
        end
        return Finding[]
    catch e
        e isa GPUC.KernelError &&
            return [Finding(string(e.message, isnothing(e.help) ? "" : '\n' * e.help))]
        e isa GPUC.InvalidIRError || rethrow()
        errors = unique(
            map(filter(!is_benign_ir_error, e.errors)) do (kind, bt, meta)
                (string(kind, isnothing(meta) ? "" : " [$meta]"), ir_error_location(bt))
            end,
        )
        return map(Base.splat(Finding), errors)
    end
end

# ─── Kernel launch extraction (compiler introspection over the host IR) ──────

"""
    KernelLaunchSite

A `CUDA.@cuda`-style launch discovered in host code by
[`extract_kernel_launches`](@ref): the kernel function type (often a closure
type constructed inside the host function), the `Tuple` type of the device
argument types, and the `"file:line"` of the launch when known. A field is
`nothing` when inference cannot determine it at the launch site (e.g. a
launch wrapper whose `cufunction` call is behind runtime dispatch); the site
is [`is_resolved`](@ref) when both types are known.
"""
struct KernelLaunchSite
    kernel_type::Union{Nothing, Type}
    arg_types::Union{Nothing, Type}
    location::Union{Nothing, String}
end

# NOTE: not a constructor method, since the default constructor (which this
# would tie with in dispatch) must not be bypassed for `Type`-valued inputs.
function launch_site(@nospecialize(F), @nospecialize(TT), location)
    kernel_type = F isa Type && Base.isdispatchtuple(Tuple{F}) ? F : nothing
    arg_types =
        TT isa DataType && TT <: Tuple && Base.isdispatchtuple(TT) ? TT :
        nothing
    return KernelLaunchSite(kernel_type, arg_types, location)
end

"Whether the launch's full kernel signature is statically known."
is_resolved(site::KernelLaunchSite) =
    !isnothing(site.kernel_type) && !isnothing(site.arg_types)

kernel_signature(site::KernelLaunchSite) =
    Tuple{site.kernel_type, site.arg_types.parameters...}

# `MethodInstance.def` is a `Module` for top-level thunks and a `Method`
# otherwise.
definition_module(mi::Core.MethodInstance) =
    mi.def isa Method ? mi.def.module : mi.def

# Best effort: the statement's own location, else the nearest preceding one,
# else the enclosing method definition (optimization can drop locations, and
# Julia versions differ in how `CodeInfo` stores them).
statement_location(code_info, i) =
    try
        j = findprev(!=(0), code_info.codelocs, i)
        entry = code_info.linetable[isnothing(j) ? 1 : code_info.codelocs[j]]
        string(entry.file, ':', entry.line)
    catch
        nothing
    end

# The type lattice element of a statement argument in optimized `code_typed`
# IR. `Expr` arguments (e.g. `:static_parameter`) are not resolved, since
# `CC.argextype` only accepts them with static-parameter types we do not have.
argument_lattice_type(code_info, @nospecialize(x)) =
    x isa Expr ? Any : CC.argextype(x, code_info, CC.VarState[])

# The constant `Type` value of a statement argument, or `nothing`.
function constant_type(code_info, @nospecialize(x))
    T = CC.widenconst(argument_lattice_type(code_info, x))
    return CC.isconstType(T) ? T.parameters[1] : nothing
end

# `CUDA.@cuda f(args...)` lowers to `kernel = cufunction(cudaconvert(f),
# Tuple{map(Core.Typeof, map(cudaconvert, args))...}); kernel(args...)`. In
# optimized IR the `cufunction` call appears either as an `:invoke` whose
# `specTypes` ends with `..., typeof(cufunction), F, Type{TT}` (possibly
# behind `Core.kwcall` or the keyword-sorter body when compiler keywords are
# passed), or — when some part of the call is not inferrable — as a dynamic
# `:call` with a `cufunction` reference among its first arguments. The kernel
# signature is `Tuple{F, TT.parameters...}`.
function match_cufunction(specTypes)
    specTypes isa DataType || return nothing
    params = (specTypes.parameters...,)
    i = findfirst(P -> P === typeof(CUDA.cufunction), params)
    (isnothing(i) || i == length(params)) && return nothing
    F = params[i + 1]
    TT = if i + 1 == length(params)
        Tuple{} # `cufunction(f)` uses the default argument types
    else
        P = params[i + 2]
        CC.isconstType(P) && P.parameters[1] isa Type ? P.parameters[1] : nothing
    end
    return F, TT
end

function match_cufunction_call(code_info, stmt)
    args = stmt.args
    # `cufunction` is the callee, or the callee argument of `Core.kwcall`.
    i = findfirst(1:min(length(args), 3)) do k
        value = CC.singleton_type(argument_lattice_type(code_info, args[k]))
        value === CUDA.cufunction
    end
    (isnothing(i) || i == length(args)) && return nothing
    F = CC.widenconst(argument_lattice_type(code_info, args[i + 1]))
    TT = i + 1 == length(args) ? Tuple{} : constant_type(code_info, args[i + 2])
    return F, TT
end

"""
    extract_kernel_launches(sig::Type{<:Tuple}; max_depth = 10, max_visits = 1000)
        -> sites::Vector{KernelLaunchSite}

Find every kernel launch reachable from a call with signature `sig` (over the
`:host` GPU argument types) by walking the optimized IR from `Base.code_typed`
and recursing into `:invoke`d callees, up to `max_depth` levels and
`max_visits` inspected methods. Launches whose kernel function type or
argument tuple type inference cannot determine (e.g. launch wrappers that
compute kernel attributes at runtime, such as a kernel-naming keyword that
makes the `cufunction` call dynamic) are returned as unresolved sites (see
[`KernelLaunchSite`](@ref)).
"""
function extract_kernel_launches(
    sig::Type{<:Tuple};
    max_depth::Integer = 10,
    max_visits::Integer = 1000,
)
    sites = KernelLaunchSite[]
    site_keys = Set{Tuple{Any, Any}}()
    visited = Set{Type}((sig,))
    queue = [(sig, 1)]
    visits = 0
    while !isempty(queue)
        (visits += 1) > max_visits && break
        (signature, depth) = popfirst!(queue)
        code_infos = try
            Base.code_typed_by_type(signature; optimize = true)
        catch
            continue # uninferrable or ambiguous signatures have no host IR
        end
        length(code_infos) == 1 || continue
        code_info = code_infos[1][1]
        code_info isa Core.CodeInfo || continue
        for (i, stmt) in enumerate(code_info.code)
            if Meta.isexpr(stmt, :invoke) && stmt.args[1] isa Core.MethodInstance
                callee = stmt.args[1]
                match = match_cufunction(callee.specTypes)
                if isnothing(match)
                    if depth < max_depth &&
                       Base.moduleroot(definition_module(callee)) !== Core &&
                       callee.specTypes isa DataType &&
                       !(callee.specTypes in visited)
                        push!(visited, callee.specTypes)
                        push!(queue, (callee.specTypes, depth + 1))
                    end
                    continue
                end
            elseif Meta.isexpr(stmt, :call)
                match = match_cufunction_call(code_info, stmt)
                isnothing(match) && continue
            else
                continue
            end
            site = launch_site(match..., statement_location(code_info, i))
            key = (site.kernel_type, site.arg_types)
            key in site_keys || (push!(site_keys, key); push!(sites, site))
        end
    end
    return sites
end

# ─── Non-isbits kernel arguments (closure captures at the launch boundary) ───

# Not every type has a definite number of fields (`Type{Float64}` does not).
definite_fieldcount(@nospecialize(T)) =
    try
        fieldcount(T)
    catch
        nothing
    end

# GPUCompiler's internal helper prints one indented line per non-isbits field,
# recursively ("  .name is of type T which is not isbits."); the fallback only
# names the immediate non-isbits fields.
explain_nonisbits(@nospecialize(T)) =
    isdefined(GPUC, :explain_nonisbits) ? GPUC.explain_nonisbits(T) :
    join(
        (
            "  .$(fieldname(T, i)) is of type $(fieldtype(T, i)) which is not isbits"
            for i in 1:something(definite_fieldcount(T), 0) if
            !isbitstype(fieldtype(T, i))
        ),
        '\n',
    )

# Mirrors GPUCompiler's `check_invocation`, which rejects these arguments when
# a kernel is compiled for a real launch ("passing non-bitstype argument").
function nonisbits_findings(site::KernelLaunchSite)
    findings = Finding[]
    for (i, T) in enumerate((site.kernel_type, site.arg_types.parameters...))
        (CC.isconstType(T) || isbitstype(T)) && continue
        definite_fieldcount(T) == 0 && continue # usable by identity
        what =
            i == 1 ? "the kernel closure `$T` captures non-isbits values" :
            "kernel argument $(i - 1) has the non-isbits type `$T`"
        detail = rstrip(explain_nonisbits(T), '\n')
        message = string(
            what,
            "; a real launch fails with \"passing non-bitstype argument\"",
            isempty(detail) ? "" : ":\n" * detail,
        )
        push!(findings, Finding(message, site.location))
    end
    return findings
end

# ─── LLVM-lowering compatibility of kernel argument types (:llvm_types) ──────

"""
    llvm_type_findings!(findings, ::Type{T}, path; llvm = Base.libllvm_version)

Append a [`Finding`](@ref) for every field or element of `T` (reached from a
kernel argument named by `path`) whose LLVM lowering is known to crash the
NVPTX backend in libLLVM version `llvm`. Each rule is gated on the LLVM
version that fixed it, so the checks disable themselves on newer Julia:

  - `Int128`/`UInt128` anywhere in a kernel argument's layout (`i128` in a
    parameter) crashes NVPTX instruction selection before LLVM 20
    (llvm/llvm-project#49221, llvm/llvm-project#83179).
  - Homogeneous `NTuple{N, <:VecElement}` SIMD vectors with non-power-of-two
    length (`<3 x i64>` etc.) crash NVPTX parameter/return lowering before
    LLVM 19 (llvm/llvm-project#104524).
  - Fully empty aggregate parameters (`{}`, `[0 x T]`) are an NVPTX fatal
    error ("Unexpected empty type") until the 2026 fix llvm/llvm-project's
    PR #207057 (first released in LLVM 23). No rule is emitted for them,
    because Julia cannot produce such parameters: empty aggregates like
    `Tuple{}` are ghost types, which GPUCompiler elides as arguments, and
    Julia's codegen elides zero-size fields inside larger aggregates
    (verified: a `Tuple{Tuple{}, Float64}` kernel argument lowers to
    `{ double }`).
"""
function llvm_type_findings!(
    findings::Vector{Finding},
    @nospecialize(T),
    path::String;
    llvm::VersionNumber = Base.libllvm_version,
    depth::Integer = 1,
)
    # `push!` returns `findings`, so `report` can be the branches' return value.
    report(problem, remedy) = push!(
        findings,
        Finding("$path is of type $T: $problem; this Julia ships LLVM $llvm. $remedy"),
    )
    depth > 32 && return findings
    if T isa Union
        for U in Base.uniontypes(T)
            llvm_type_findings!(findings, U, path; llvm, depth = depth + 1)
        end
        return findings
    end
    T isa DataType || return findings
    (T === Int128 || T === UInt128) && llvm < v"20" &&
        return report(
            "an `i128` in a kernel parameter crashes NVPTX instruction selection \
             before LLVM 20 (llvm/llvm-project#49221, llvm/llvm-project#83179)",
            "Pass the value as an `NTuple` of smaller integers and reassemble it \
             inside the kernel.",
        )
    if T <: Tuple &&
       length(T.parameters) > 1 &&
       !ispow2(length(T.parameters)) &&
       allequal(T.parameters) &&
       first(T.parameters) isa DataType &&
       first(T.parameters) <: VecElement &&
       llvm < v"19"
        return report(
            "a SIMD vector with a non-power-of-two number of elements crashes \
             NVPTX parameter/return lowering before LLVM 19 \
             (llvm/llvm-project#104524)",
            "Pad the vector to a power-of-two length.",
        )
    end
    (isprimitivetype(T) || !isconcretetype(T)) && return findings
    for i in 1:fieldcount(T)
        name = T <: Tuple ? "[$i]" : ".$(fieldname(T, i))"
        llvm_type_findings!(findings, fieldtype(T, i), path * name; llvm, depth = depth + 1)
    end
    return findings
end

function llvm_argument_findings!(findings, arg_types, path_prefix; llvm)
    for (i, T) in enumerate(arg_types)
        Base.issingletontype(T) && continue # elided by GPUCompiler, never lowered
        llvm_type_findings!(findings, T, "$path_prefix[$i]"; llvm)
    end
    return findings
end

# ─── Host pointer scan (for kernel arguments after adaptation) ───────────────

function host_pointer_findings!(findings, x, path)
    if x isa Array || x isa Ptr
        push!(
            findings,
            Finding(
                "$path is a host $(typeof(x).name.wrapper), which would cause \
                 an illegal memory access in a kernel",
            ),
        )
    elseif x isa CUDA.CuDeviceArray
        return findings
    elseif !isbits(x) && (isstructtype(typeof(x)) || x isa Tuple)
        foreach(1:fieldcount(typeof(x))) do i
            name = x isa Tuple ? "[$i]" : string('.', fieldname(typeof(x), i))
            isdefined(x, i) &&
                host_pointer_findings!(findings, getfield(x, i), path * name)
        end
    end
    return findings
end

# ─── Combined checks ─────────────────────────────────────────────────────────

add_issue_report!(reports, stage::Symbol, label, findings) =
    isempty(findings) || push!(reports, IssueStageReport(stage, label, findings))

# GPUCompiler IR validation and JET analysis over the device method table, for
# one kernel signature (an extracted launch or the whole-call fallback).
function add_kernel_analyses!(reports, sig, label; jetconfigs...)
    add_issue_report!(
        reports,
        :kernel,
        "GPUCompiler IR validation ($label)",
        ir_findings(sig),
    )
    return add_jet_report!(
        reports,
        :kernel,
        "JET over CUDA's device method table ($label)",
        sig;
        device = true,
        jetconfigs...,
    )
end

"""
    compilation_reports(f, args::Tuple; stages, kwargs...)
        -> (ok::Bool, reports::Vector{StageReport})

Run the [`TestCompilation`](@ref) analyses of `f(args...)` selected by
`stages` (any subset of `$(DEFAULT_STAGES_DOC)`; all by default). `ok` is
`true` when every selected analysis finds no issues, and `reports` contains
one [`StageReport`](@ref) per failing analysis. See the module docstring for
the available stages and keyword arguments.
"""
function compilation_reports(
    f,
    args::Tuple;
    stages = DEFAULT_STAGES,
    host_array_type::Type = CUDA.CuArray,
    type_replacements::Tuple = (),
    host_ignored_modules = default_host_ignored_modules(),
    kernel_ignored_modules = (),
    extra_ignored_modules = (),
    max_extraction_depth::Integer = 10,
    max_extraction_visits::Integer = 1000,
    jetconfigs...,
)
    for stage in stages
        stage in DEFAULT_STAGES ||
            throw(ArgumentError("unknown stage $stage; expected one of \
                                 $DEFAULT_STAGES_DOC"))
    end
    host_configs = (;
        ignored_modules = (host_ignored_modules..., extra_ignored_modules...),
        jetconfigs...,
    )
    kernel_configs = (;
        ignored_modules = (kernel_ignored_modules..., extra_ignored_modules...),
        jetconfigs...,
    )
    reports = StageReport[]

    :cpu in stages && add_jet_report!(
        reports,
        :cpu,
        "JET optimization analysis over CPU argument types",
        Tuple{typeof(f), map(typeof, args)...};
        host_configs...,
    )

    # The GPU-machine call signature, used by :host and by launch extraction.
    host_sig = nothing
    if :host in stages || :kernel in stages || :llvm_types in stages
        host_sig, host_findings =
            host_signature(f, args; host_array_type, type_replacements)
        add_issue_report!(
            reports,
            # Attribute the failure to the first stage that needs the types.
            first(filter(in(stages), (:host, :kernel, :llvm_types))),
            "GPU host argument types (launch extraction skipped)",
            host_findings,
        )
    end
    :host in stages &&
        !isnothing(host_sig) &&
        add_jet_report!(
            reports,
            :host,
            "JET optimization analysis over GPU host argument types",
            host_sig;
            host_configs...,
        )

    (:kernel in stages || :pointers in stages || :llvm_types in stages) ||
        return isempty(reports), reports

    adapted = kernel_arguments(args)

    if :pointers in stages
        findings = Finding[]
        foreach(enumerate(adapted)) do (i, arg)
            host_pointer_findings!(findings, arg, "args[$i]")
        end
        add_issue_report!(reports, :pointers, "host pointers after adaptation", findings)
    end

    sites =
        (:kernel in stages || :llvm_types in stages) && !isnothing(host_sig) ?
        extract_kernel_launches(
            host_sig;
            max_depth = max_extraction_depth,
            max_visits = max_extraction_visits,
        ) : KernelLaunchSite[]
    launches = filter(is_resolved, sites)

    if :kernel in stages && isempty(launches)
        # Whole-call fallback: treat f(adapted args...) as one kernel body.
        kernel_f = (kernel_args...) -> (f(kernel_args...); nothing)
        sig = Tuple{typeof(kernel_f), map(typeof, adapted)...}
        add_kernel_analyses!(reports, sig, "whole call as kernel body"; kernel_configs...)
    elseif :kernel in stages
        for (k, site) in enumerate(launches)
            label = string(
                "kernel $k: ",
                site.kernel_type,
                isnothing(site.location) ? "" : " launched at $(site.location)",
            )
            capture_findings = nonisbits_findings(site)
            if isempty(capture_findings)
                add_kernel_analyses!(
                    reports,
                    kernel_signature(site),
                    label;
                    kernel_configs...,
                )
            else # the launch (and thus compilation) cannot happen
                add_issue_report!(
                    reports,
                    :kernel,
                    "launch argument validation ($label)",
                    capture_findings,
                )
            end
        end
    end

    if :llvm_types in stages
        llvm = Base.libllvm_version
        findings = Finding[]
        llvm_argument_findings!(findings, map(typeof, adapted), "args"; llvm)
        # Unresolved sites with known argument types still contribute here.
        for (k, site) in enumerate(sites)
            isnothing(site.kernel_type) ||
                llvm_type_findings!(findings, site.kernel_type, "kernel $k closure"; llvm)
            isnothing(site.arg_types) || llvm_argument_findings!(
                findings,
                (site.arg_types.parameters...,),
                "kernel $k args";
                llvm,
            )
        end
        unique!(finding -> finding.message, findings)
        add_issue_report!(
            reports,
            :llvm_types,
            "LLVM lowering of kernel argument types (libLLVM $llvm)",
            findings,
        )
    end

    return isempty(reports), reports
end

# ─── Test integration ────────────────────────────────────────────────────────

"""
    CompilationTestFailure

The `Test.Result` recorded by a failing [`@test_compilation`](@ref), holding
every failing [`StageReport`](@ref). Its `show` method prints one organized,
color-coded report in the style of `JET.@test_opt` failures: the source
location and tested expression, then each stage's findings under a stage
header (JET results are printed with JET's own report printer).
"""
struct CompilationTestFailure <: Test.Result
    orig_expr::Expr
    source::LineNumberNode
    reports::Vector{StageReport}
end

function Base.show(io::IO, t::CompilationTestFailure)
    printstyled(io, "Compilation test failed"; bold = true, color = Base.error_color())
    print(io, " at ")
    printstyled(io, something(t.source.file, :none), ":", t.source.line, "\n"; bold = true)
    println(io, "  Expression: ", t.orig_expr)
    for report in t.reports
        println(io, indent_lines(sprint(show, report; context = io), "  "))
    end
end

function Test.record(::Test.FallbackTestSet, t::CompilationTestFailure)
    println(t)
    throw(Test.FallbackTestSetException("There was an error during testing"))
end

function Test.record(ts::Test.DefaultTestSet, t::CompilationTestFailure)
    if Test.TESTSET_PRINT_ENABLE[]
        printstyled(ts.description, ": "; color = :white)
        print(t)
        println()
    end
    # Convert to `Fail` so that test summarization works correctly (the same
    # approach as JET's `Test.record(::DefaultTestSet, ::JETTestFailure)`).
    push!(ts.results, Test.Fail(:test_compilation, t.orig_expr, nothing, nothing, t.source))
    return t
end

"""
    @test_compilation [stages = ...] [kwarg = ...] f(args...)

Assert that `f(args...)` passes the selected [`compilation_reports`](@ref)
analyses, recording the result in the enclosing `Test` test set. On failure,
one [`CompilationTestFailure`](@ref) is recorded, which prints an organized
report of every failing stage's findings.

    @test_compilation fill!(data, value)
    @test_compilation stages = (:cpu, :host) my_solver_step!(state, cache)
"""
macro test_compilation(args...)
    call = args[end]
    kwargs = map(args[1:(end - 1)]) do kwarg
        @assert Meta.isexpr(kwarg, :(=), 2) "expected keyword arguments before the call"
        Expr(:kw, kwarg.args[1], esc(kwarg.args[2]))
    end
    @assert Meta.isexpr(call, :call) "expected a function call as the last argument"
    f = esc(call.args[1])
    call_args = map(esc, call.args[2:end])
    orig_expr = QuoteNode(
        Expr(:macrocall, Symbol("@test_compilation"), nothing, args...),
    )
    source = QuoteNode(__source__)
    return quote
        testres = try
            (ok, reports) =
                compilation_reports($f, ($(call_args...),); $(kwargs...))
            if ok
                Test.Pass(:test_compilation, $orig_expr, nothing, nothing, $source)
            else
                CompilationTestFailure($orig_expr, $source, reports)
            end
        catch err
            err isa InterruptException && rethrow()
            Test.Error(:test_error, $orig_expr, err, Base.current_exceptions(), $source)
        end
        Test.record(Test.get_testset(), testres)
    end
end

end # module TestCompilation
