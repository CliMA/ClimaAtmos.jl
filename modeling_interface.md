#### Hey! Let's talk about interfaces.

## Introduction

The ClimaAtmos interface can be split into two components: the *modeling interface* and the *driver interface* (this could also be called the *simulation interface*, in the parlance of Oceananigans). Roughly speaking, the modeling interface allows one to specify a PDE, while the driver interface allows one to specify how to initialize a state and evolve it according to that PDE. Specifically, the modeling interface determines how an arbitrary initial condition should change over time; it is the language through which one controls the explicit and implicit tendencies of a simulated system, as well as the Jacobian approximation of the implicit tendency. The driver interface encompasses everything else that the user needs to control—the domain over which the PDE is being solved, how the domain is discretized, the initial state of the system over this discretized domain, the time period over which the state must be integrated, the details of this integration process, and the post-processing that occurs after the final state is obtained.

This document only focuses on the modeling interface. The requirements of the driver interface are still evolving, so it is not yet possible to write a detailed design document for the driver interface. Moreover, the driver interface will serve as a wrapper for the modeling interface, so it is a good idea to finalize the modeling interface before discussing the driver interface.

This document is organized as follows. We begin with a broad overview of the original ClimaAtmos modeling interface, followed by a discussion of its high-level design issues. We then develop a new interface that fixes these issues, starting from scratch and justifying every major design decision. Finally, we provide some examples of what code that uses this new interface will look like.

## The Original ClimaAtmos Interface

The top-level data structure in the original modeling interface is the `AbstractModel`, which currently has 3 implementations—`Nonhydrostatic3DModel`, `Nonhydrostatic2DModel`, and `SingleColumnModel`. Let's focus on the first one:
```julia
Base.@kwdef struct Nonhydrostatic3DModel{D, B, T, M, VD, F, BC, P, FT, C} <:
                   AbstractNonhydrostatic3DModel
    domain::D
    base::B = AdvectiveForm()
    thermodynamics::T = TotalEnergy()
    moisture::M = Dry()
    vertical_diffusion::VD = NoVerticalDiffusion()
    flux_corr::F = true
    hyperdiffusivity::FT
    boundary_conditions::BC
    parameters::P
    cache::C = CacheEmpty()
end
```
The majority of the fields in this data structure are *flags*, which toggle between different terms in the tendency. A flag could be a primitive datatype, like the boolean `flux_corr`, which determines whether to add flux correction to the tendency. Similarly, it could be a symbol (e.g., `thermodynamics` could have been set to `:total_energy` instead of `TotalEnergy()`). In Julia, it is more common to turn flags into *singleton objects*—objects that have no fields, but whose types encode the value of the flag. If such a flag is passed as an argument to a function, that function can have several different methods, each corresponding to a different type of the flag. Essentially, this design pattern allows the compiler to know the value of the flag, so that it does not need to look up the flag's value at runtime to decide which code to run.

Aside from flags, this data structure also contains *parameters*, which are numerical values used in various tendency terms. In this case, there is a single number `hyperdiffusivity`, as well as a collection of numbers encoded in `parameters`, which is a `CLIMAParameters.AbstractEarthParameterSet`.

In addition, this data structure contains the boundary conditions of the PDE. Unfortunately, the ability to use non-trivial boundary conditions was never implemented in this interface. The next section will outline why this feature would have been rather difficult to implement.

The remaining fields in this data structure are `domain` and `cache`. `domain` is a description of the space over which the simulated state is defined; this is not used by the data structure, and would probably have been removed had development of this interface continued. `cache` is a collection of cached variables used to speed up the computation of a tendency. Like with boundary conditions, the ability to use a non-empty cache was never implemented, and the next section will outline why. Instead, cached variables were reallocated on every tendency evaluation.

The main function used for this data structure is `make_ode_function(model::Nonhydrostatic3DModel)`, which returns a function that computes the explicit tendency for the simulated system. If development of this interface had continued, this might have been generalized for IMEX methods by having `make_ode_function` return a `SplitODEFunction` object, which would store functions that compute the explicit and implicit tendencies, as well as a function that computes the Jacobian of the implicit tendency.

## Why We Needed A New Interface
