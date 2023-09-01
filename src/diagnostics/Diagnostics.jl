# Diagnostics.jl
#
# This file contains:
#
# - The definition of what a DiagnosticVariable is. Morally, a DiagnosticVariable is a
#   variable we know how to compute from the state. We attach more information to it for
#   documentation and to reference to it with its long name. DiagnosticVariables can exist
#   irrespective of the existence of an actual simulation that is being run. ClimaAtmos
#   comes with several diagnostics already defined (in the `ALL_DIAGNOSTICS` dictionary).
#
# - A dictionary `ALL_DIAGNOSTICS` with all the diagnostics we know how to compute, keyed
#   over their long name. If you want to add more diagnostics, look at the included files.
#   You can add your own file if you want to define several new diagnostics that are
#   conceptually related.
#
# - The definition of what a ScheduledDiagnostics is. Morally, a ScheduledDiagnostics is a
#   DiagnosticVariable we want to compute in a given simulation. For example, it could be
#   the temperature averaged over a day. We can have multiple ScheduledDiagnostics for the
#   same DiagnosticVariable (e.g., daily and monthly average temperatures).
#
#   We provide two types of ScheduledDiagnostics: ScheduledDiagnosticIterations and
#   ScheduledDiagnosticTime, with the difference being only in what domain the recurrence
#   time is defined (are we doing something at every N timesteps or every T seconds?). It is
#   much cleaner and simpler to work with ScheduledDiagnosticIterations because iterations
#   are well defined and consistent. On the other hand, working in the time domain requires
#   dealing with what happens when the timestep is not lined up with the output period.
#   Possible solutions to this problem include: uneven output, interpolation, or restricting
#   the user from picking specific combinations of timestep/output period. In the current
#   implementation, we choose the third option. So, ScheduledDiagnosticTime is provided
#   because it is the physically interesting quantity. If we know what is the timestep, we
#   can convert between the two and check if the diagnostics are well-posed in terms of the
#   relationship between the periods and the timesteps. In some sense, you can think of
#   ScheduledDiagnosticIterations as an internal representation and ScheduledDiagnosticTime
#   as the external interface.
#
# - A function to convert a list of ScheduledDiagnosticIterations into a list of
#   AtmosCallbacks. This function takes three arguments: the list of diagnostics and two
#   dictionaries that map each scheduled diagnostic to an area of memory where to save the
#   result and where to keep track of how many times the function was called (so that we
#   can compute stuff like averages).
#
# - This file also also include several other files, including (but not limited to):
#   - core_diagnostics.jl
#   - reduction_identities.jl

"""
    DiagnosticVariable


A recipe to compute a diagnostic variable from the state, along with some useful metadata.

The primary use for `DiagnosticVariable`s is to be embedded in a `ScheduledDiagnostic` to
compute diagnostics while the simulation is running.

The metadata is used exclusively by the `output_writer` in the `ScheduledDiagnostic`. It is
responsibility of the `output_writer` to follow the conventions about the meaning of the
metadata and their use.

In `ClimaAtmos`, we roughly follow the naming conventions listed in this file:
https://docs.google.com/spreadsheets/d/1qUauozwXkq7r1g-L4ALMIkCNINIhhCPx

Keyword arguments
=================

- `short_name`: Name used to identify the variable in the output files and in the file
                names. Short but descriptive. `ClimaAtmos` follows the CMIP conventions and
                the diagnostics are identified by the short name.

- `long_name`: Name used to identify the variable in the input files.

- `units`: Physical units of the variable.

- `comments`: More verbose explanation of what the variable is, or comments related to how
              it is defined or computed.

- `compute_from_integrator`: Function that compute the diagnostic variable from the state.
                             It has to take two arguments: the `integrator`, and a
                             pre-allocated area of memory where to write the result of the
                             computation. It the no pre-allocated area is available, a new
                             one will be allocated. To avoid extra allocations, this
                             function should perform the calculation in-place (i.e., using
                             `.=`).
"""
Base.@kwdef struct DiagnosticVariable{T <: AbstractString, T2}
    short_name::T
    long_name::T
    units::T
    comments::T
    compute_from_integrator::T2
end

# ClimaAtmos diagnostics

const ALL_DIAGNOSTICS = Dict{String, DiagnosticVariable}()

"""

    add_diagnostic_variable!(; short_name,
                               long_name,
                               units,
                               description,
                               compute_from_integrator)


Add a new variable to the `ALL_DIAGNOSTICS` dictionary (this function mutates the state of
`ClimaAtmos.ALL_DIAGNOSTICS`).

If possible, please follow the naming scheme outline in
https://docs.google.com/spreadsheets/d/1qUauozwXkq7r1g-L4ALMIkCNINIhhCPx

Keyword arguments
=================


- `short_name`: Name used to identify the variable in the output files and in the file
                names. Short but descriptive. `ClimaAtmos` diagnostics are identified by the
                short name. We follow the Coupled Model Intercomparison Project conventions.

- `long_name`: Name used to identify the variable in the input files.

- `units`: Physical units of the variable.

- `comments`: More verbose explanation of what the variable is, or comments related to how
              it is defined or computed.

- `compute_from_integrator`: Function that compute the diagnostic variable from the state.
                             It has to take two arguments: the `integrator`, and a
                             pre-allocated area of memory where to write the result of the
                             computation. It the no pre-allocated area is available, a new
                             one will be allocated. To avoid extra allocations, this
                             function should perform the calculation in-place (i.e., using
                             `.=`).

"""
function add_diagnostic_variable!(;
    short_name,
    long_name,
    units,
    comments,
    compute_from_integrator
)
    haskey(ALL_DIAGNOSTICS, short_name) &&
        error("diagnostic $short_name already defined")

    ALL_DIAGNOSTICS[short_name] = DiagnosticVariable(;
        short_name,
        long_name,
        units,
        comments,
        compute_from_integrator
    )
end

# Do you want to define more diagnostics? Add them here
include("core_diagnostics.jl")


# ScheduledDiagnostics

# NOTE: The definitions of ScheduledDiagnosticTime and ScheduledDiagnosticIterations are
# nearly identical except for the fact that one is assumed to use units of seconds the other
# units of integration steps. However, we allow for this little repetition of code to avoid
# adding an extra layer of abstraction just to deal with these two objects (some people say
# that "duplication is better than over-abstraction"). Most users will only work with
# ScheduledDiagnosticTime. (It would be nice to have defaults fields in abstract types, as
# proposed in 2013 in https://github.com/JuliaLang/julia/issues/4935) Having two distinct
# types allow us to implement different checks and behaviors (e.g., we allow
# ScheduledDiagnosticTime to have placeholders values for {compute, output}_every so that we
# can plug the timestep in it).

struct ScheduledDiagnosticIterations{T1, T2, OW, F1, F2, PO}
    variable::DiagnosticVariable
    output_every::T1
    output_writer::OW
    reduction_time_func::F1
    reduction_space_func::F2
    compute_every::T2
    pre_output_hook!::PO

    """
        ScheduledDiagnosticIterations(; variable::DiagnosticVariable,
                                        output_every,
                                        output_writer,
                                        reduction_time_func = nothing,
                                        reduction_space_func = nothing,
                                        compute_every = isa_reduction ? 1 : output_every,
                                        pre_output_hook! = (accum, count) -> nothing)


    A `DiagnosticVariable` that has to be computed and output during a simulation with a cadence
    defined by the number of iterations, with an optional reduction applied to it (e.g., compute
    the maximum temperature over the course of every 10 timesteps). This object is turned into
    two callbacks (one for computing and the other for output) and executed by the integrator.

    Keyword arguments
    =================

    - `variable`: The diagnostic variable that has to be computed and output.

    - `output_every`: Save the results to disk every `output_every` iterations.

    - `output_writer`: Function that controls out to save the computed diagnostic variable to
                       disk. `output_writer` has to take three arguments: the value that has to
                       be output, the `ScheduledDiagnostic`, and the integrator. Internally, the
                       integrator contains extra information (such as the current timestep). It
                       is responsibility of the `output_writer` to properly use the provided
                       information for meaningful output.

    - `reduction_time_func`: If not `nothing`, this `ScheduledDiagnostic` receives an area of
                             scratch space `acc` where to accumulate partial results. Then, at
                             every `compute_every`, `reduction_time_func` is computed between
                             the previously stored value in `acc` and the new value. This
                             implements a running reduction. For example, if
                             `reduction_time_func = max`, the space `acc` will hold the running
                             maxima of the diagnostic. To implement operations like the
                             arithmetic average, the `reduction_time_func` has to be chosen as
                             `sum`, and a `pre_output_hook!` that renormalize `acc` by the
                             number of samples has to be provided. For custom reductions, it is
                             necessary to also specify the identity of operation by defining a
                             new method to `identity_of_reduction`.

    - `reduction_space_func`: NOT IMPLEMENTED YET

    - `compute_every`: Run the computations every `compute_every` iterations. This is not
                       particularly useful for point-wise diagnostics, where we enforce that
                       `compute_every` = `output_every`. For time reductions, `compute_every` is
                       set to 1 (compute at every timestep) by default. `compute_every` has to
                       evenly divide `output_every`.

    - `pre_output_hook!`: Function that has to be run before saving to disk for reductions
                          (mostly used to implement averages). The function `pre_output_hook!`
                          is called with two arguments: the value accumulated during the
                          reduction, and the number of times the diagnostic was computed from
                          the last time it was output. `pre_output_hook!` should mutate the
                          accumulator in place. The return value of `pre_output_hook!` is
                          discarded. An example of `pre_output_hook!` to compute the arithmetic
                          average is `pre_output_hook!(acc, N) = @. acc = acc / N`.

    """
    function ScheduledDiagnosticIterations(;
        variable::DiagnosticVariable,
        output_every,
        output_writer,
        reduction_time_func = nothing,
        reduction_space_func = nothing,
        compute_every = isnothing(reduction_time_func) ? output_every : 1,
        pre_output_hook! = (accum, count) -> nothing,
    )

        # We provide an inner constructor to enforce some constraints

        output_every % compute_every == 0 || error(
            "output_every should be multiple of compute_every for variable $(variable.long_name)",
        )

        isa_reduction = !isnothing(reduction_time_func)

        # If it is not a reduction, we compute only when we output
        if !isa_reduction && compute_every != output_every
            @warn "output_every != compute_every for $(variable.long_name), changing compute_every to match"
            compute_every = output_every
        end

        T1 = typeof(output_every)
        T2 = typeof(compute_every)
        OW = typeof(output_writer)
        F1 = typeof(reduction_time_func)
        F2 = typeof(reduction_space_func)
        PO = typeof(pre_output_hook!)

        new{T1, T2, OW, F1, F2, PO}(
            variable,
            output_every,
            output_writer,
            reduction_time_func,
            reduction_space_func,
            compute_every,
            pre_output_hook!,
        )
    end
end


struct ScheduledDiagnosticTime{T1, T2, OW, F1, F2, PO}
    variable::DiagnosticVariable
    output_every::T1
    output_writer::OW
    reduction_time_func::F1
    reduction_space_func::F2
    compute_every::T2
    pre_output_hook!::PO

    """
        ScheduledDiagnosticTime(; variable::DiagnosticVariable,
                                            output_every,
                                            output_writer,
                                            reduction_time_func = nothing,
                                            reduction_space_func = nothing,
                                            compute_every = isa_reduction ? :timestep : output_every,
                                            pre_output_hook! = (accum, count) -> nothing)


    A `DiagnosticVariable` that has to be computed and output during a simulation with a
    cadence defined by how many seconds in simulation time, with an optional reduction
    applied to it (e.g., compute the maximum temperature over the course of every day). This
    object is turned into a `ScheduledDiagnosticIterations`, which is turned into two
    callbacks (one for computing and the other for output) and executed by the integrator.

    Keyword arguments
    =================

    - `variable`: The diagnostic variable that has to be computed and output.

    - `output_every`: Save the results to disk every `output_every` seconds.

    - `output_writer`: Function that controls out to save the computed diagnostic variable to
                       disk. `output_writer` has to take three arguments: the value that has to
                       be output, the `ScheduledDiagnostic`, and the integrator. Internally, the
                       integrator contains extra information (such as the current timestep). It
                       is responsibility of the `output_writer` to properly use the provided
                       information for meaningful output.

    - `reduction_time_func`: If not `nothing`, this `ScheduledDiagnostic` receives an area of
                             scratch space `acc` where to accumulate partial results. Then, at
                             every `compute_every`, `reduction_time_func` is computed between
                             the previously stored value in `acc` and the new value. This
                             implements a running reduction. For example, if
                             `reduction_time_func = max`, the space `acc` will hold the running
                             maxima of the diagnostic. To implement operations like the
                             arithmetic average, the `reduction_time_func` has to be chosen as
                             `sum`, and a `pre_output_hook!` that renormalize `acc` by the
                             number of samples has to be provided. For custom reductions, it is
                             necessary to also specify the identity of operation by defining a
                             new method to `identity_of_reduction`.

    - `reduction_space_func`: NOT IMPLEMENTED YET

    - `compute_every`: Run the computations every `compute_every` seconds. This is not
                       particularly useful for point-wise diagnostics, where we enforce that
                       `compute_every` = `output_every`. For time reductions,
                       `compute_every` is set to `:timestep` (compute at every timestep) by
                       default. `compute_every` has to evenly divide `output_every`.
                       `compute_every` can take the special symbol `:timestep` which is a
                       placeholder for the timestep of the simulation to which this
                       `ScheduledDiagnostic` is attached.

    - `pre_output_hook!`: Function that has to be run before saving to disk for reductions
                          (mostly used to implement averages). The function `pre_output_hook!`
                          is called with two arguments: the value accumulated during the
                          reduction, and the number of times the diagnostic was computed from
                          the last time it was output. `pre_output_hook!` should mutate the
                          accumulator in place. The return value of `pre_output_hook!` is
                          discarded. An example of `pre_output_hook!` to compute the arithmetic
                          average is `pre_output_hook!(acc, N) = @. acc = acc / N`.

    """
    function ScheduledDiagnosticTime(;
        variable::DiagnosticVariable,
        output_every,
        output_writer,
        reduction_time_func = nothing,
        reduction_space_func = nothing,
        compute_every = isnothing(reduction_time_func) ? output_every :
                        :timestep,
        pre_output_hook! = (accum, count) -> nothing,
    )

        # We provide an inner constructor to enforce some constraints

        # compute_every could be a Symbol (:timestep). We process this that when we process
        # the list of diagnostics
        if !isa(compute_every, Symbol)
            output_every % compute_every == 0 || error(
                "output_every should be multiple of compute_every for variable $(variable.long_name)",
            )
        end

        isa_reduction = !isnothing(reduction_time_func)

        # If it is not a reduction, we compute only when we output
        if !isa_reduction && compute_every != output_every
            @warn "output_every != compute_every for $(variable.long_name), changing compute_every to match"
            compute_every = output_every
        end

        T1 = typeof(output_every)
        T2 = typeof(compute_every)
        OW = typeof(output_writer)
        F1 = typeof(reduction_time_func)
        F2 = typeof(reduction_space_func)
        PO = typeof(pre_output_hook!)

        new{T1, T2, OW, F1, F2, PO}(
            variable,
            output_every,
            output_writer,
            reduction_time_func,
            reduction_space_func,
            compute_every,
            pre_output_hook!,
        )
    end
end

"""
    ScheduledDiagnosticIterations(sd_time::ScheduledDiagnosticTime, Δt)


Create a `ScheduledDiagnosticIterations` given a `ScheduledDiagnosticTime` and a timestep
`Δt`. In this, ensure that `compute_every` and `output_every` are meaningful for the given
timestep.

"""

function ScheduledDiagnosticIterations(
    sd_time::ScheduledDiagnosticTime,
    Δt::T,
) where {T}

    # If we have the timestep, we can convert time in seconds into iterations

    # if compute_every is :timestep, then we want to compute after every iterations
    compute_every =
        sd_time.compute_every == :timestep ? 1 : sd_time.compute_every / Δt
    output_every = sd_time.output_every / Δt

    isinteger(output_every) || error(
        "output_every should be multiple of the timestep for variable $(sd_time.variable.long_name)",
    )
    isinteger(compute_every) || error(
        "compute_every should be multiple of the timestep for variable $(sd_time.variable.long_name)",
    )

    ScheduledDiagnosticIterations(;
        sd_time.variable,
        output_every = convert(Int, output_every),
        sd_time.output_writer,
        sd_time.reduction_time_func,
        sd_time.reduction_space_func,
        compute_every = convert(Int, compute_every),
        sd_time.pre_output_hook!,
    )
end

# We provide also a companion constructor for ScheduledDiagnosticIterations which returns
# itself (without copy) when called with a timestep.
#
# This is so that we can assume that
# ScheduledDiagnosticIterations(ScheduledDiagnostic{Time, Iterations}, Δt)
# always returns a valid ScheduledDiagnosticIterations
ScheduledDiagnosticIterations(
    sd::ScheduledDiagnosticIterations,
    _Δt::T,
) where {T} = sd

# We define all the known identities in reduction_identities.jl
include("reduction_identities.jl")



"""
    get_callbacks_from_diagnostics(diagnostics, storage, counters)


Translate a list of diagnostics into a list of callbacks.

Positional arguments
=====================

- `diagnostics`: List of `ScheduledDiagnosticIterations` that have to be converted to
                 callbacks. We want to have `ScheduledDiagnosticIterations` here so that we
                 can define callbacks that occur at the end of every N integration steps.

- `storage`: Dictionary that maps a given `ScheduledDiagnosticIterations` to a potentially
             pre-allocated area of memory where to accumulate/save results.

- `counters`: Dictionary that maps a given `ScheduledDiagnosticIterations` to the counter
              that tracks how many times the given diagnostics was computed from the last
              time it was output to disk.

"""
function get_callbacks_from_diagnostics(diagnostics, storage, counters)
    # We have two types of callbacks: to compute and accumulate diagnostics, and to dump
    # them to disk. Note that our callbacks do not contain any branching

    # storage is used to pre-allocate memory and to accumulate partial results for those
    # diagnostics that perform reductions.

    callbacks = Any[]

    for diag in diagnostics
        variable = diag.variable
        isa_reduction = !isnothing(diag.reduction_time_func)

        # reduction is used below. If we are not given a reduction_time_func, we just want
        # to move the computed quantity to its storage (so, we return the second argument,
        # which that will be the newly computed one). If we have a reduction, we apply it
        # point-wise
        reduction = isa_reduction ? diag.reduction_time_func : (_, y) -> y

        # If we have a reduction, we have to reset the accumulator to its neutral state. (If
        # we don't have a reduction, we don't have to do anything)
        #
        # ClimaAtmos defines methods for identity_of_reduction for standard
        # reduction_time_func in reduction_identities.jl
        reset_accumulator! =
            isa_reduction ?
            () -> begin
                # identity_of_reduction works by dispatching over Val{operation}
                identity =
                    identity_of_reduction(Val(diag.reduction_time_func))
                # We also need to make sure that we are consistent with the types
                float_type = eltype(storage[diag])
                identity_ft = convert(float_type, identity)
                storage[diag] .= identity_ft
            end : () -> nothing

        compute_callback =
            integrator -> begin
                # FIXME: Change when ClimaCore overrides .= for us to avoid multiple allocations
                value = variable.compute_from_integrator(integrator, nothing)
                storage[diag] .= reduction.(storage[diag], value)
                counters[diag] += 1
                return nothing
            end

        output_callback =
            integrator -> begin
                # Any operations we have to perform before writing to output?
                # Here is where we would divide by N to obtain an arithmetic average
                diag.pre_output_hook!(storage[diag], counters[diag])

                # Write to disk
                diag.output_writer(storage[diag], diag, integrator)

                reset_accumulator!()
                counters[diag] = 0
                return nothing
            end

        # Here we have skip_first = true. This is important because we are going to manually
        # call all the callbacks so that we can verify that they are meaningful for the
        # model under consideration (and they don't have bugs).
        append!(
            callbacks,
            [
                call_every_n_steps(
                    compute_callback,
                    diag.compute_every,
                    skip_first = true,
                ),
                call_every_n_steps(
                    output_callback,
                    diag.output_every,
                    skip_first = true,
                ),
            ],
        )
    end

    return callbacks
end
