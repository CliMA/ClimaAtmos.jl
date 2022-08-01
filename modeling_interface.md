#### Hey! Let's talk about interfaces.

## Introduction

The ClimaAtmos interface can be split into two components: the *modeling interface* and the *driver interface* (the latter could also be called the *simulation interface*, in the parlance of Oceananigans). Roughly speaking, the modeling interface allows the user to specify a time-dependent PDE, while the driver interface allows the user to specify how to initialize a state and evolve it according to that PDE. Specifically, the modeling interface is the language through which the user controls how an arbitrary state of the simulated system should change over time, which we refer to as the *total tendency* of the system. To describe the total tendency, the user must specify the individual tendencies that are in it, whether each of these tendencies is treated explicitly or implicitly, and how the Jacobians of the implicit tendencies are computed, and they must also specify the boundary conditions of the system. The driver interface encompasses everything else that the user needs to control—the domain over which the PDE is being solved, how the domain is discretized, the initial state of the system over this discretized domain, the time period over which the state must be integrated, the details of this integration process, and the post-processing that occurs after the final state is obtained.

This document only focuses on the modeling interface. The requirements of the driver interface are still evolving, so it is not yet possible to write a detailed design document for the driver interface. Moreover, the driver interface will serve as a wrapper for the modeling interface, so it is a good idea to finalize the modeling interface before moving on to the driver interface.

This document is organized as follows. We begin with a quick overview of the original ClimaAtmos modeling interface, followed by a discussion of its high-level design issues. We then develop a new interface that fixes these issues, starting from scratch and justifying every major design decision. Finally, we provide some examples of what code that uses this new interface will look like.

## The Original ClimaAtmos Interface

The main data structure in the original modeling interface is the `AbstractModel`, which currently has 3 implementations—`Nonhydrostatic3DModel`, `Nonhydrostatic2DModel`, and `SingleColumnModel`. Let's focus on the first one:
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
Half of the fields in this data structure are *flags*, which allow the user to specify which tendencies are included in the total tendency. A flag could be a primitive datatype, like the boolean `flux_corr`, which determines whether to add flux correction to the tendency. Similarly, it could be a symbol (e.g., `thermodynamics` could have been set to `:total_energy` instead of `TotalEnergy()`). In Julia, it is more common to turn flags into *singleton objects*—objects that have no fields, but whose types encode the value of the flag. If such a flag is passed as an argument to a function, that function can have several different methods, each corresponding to a different type of the flag. Essentially, this design pattern allows the compiler to know the value of the flag, so that it does not need to look up the flag's value at runtime to decide which code to run.

Aside from flags, this data structure also contains *parameters*, which are numerical values used in various tendencies. In this case, there is a single number `hyperdiffusivity`, as well as a collection of numbers encoded in `parameters`, which is a `CLIMAParameters.AbstractEarthParameterSet`.

In addition, this data structure contains the boundary conditions of the PDE. Unfortunately, the ability to use non-trivial boundary conditions was never implemented in this interface. The next section will outline why this feature would have been rather difficult to implement.

The remaining fields in this data structure are `domain` and `cache`. `domain` is a description of the space over which the simulated state is defined; this is not used by the data structure, and would probably have been removed had development of this interface continued. `cache` is a collection of cached variables used to speed up the computation of a tendency by ensuring that values are not computed more than once. Like with boundary conditions, the ability to use a non-empty cache was never implemented, and the next section will outline why. Instead, in this interface, variables that would have been stored in the cache get reallocated on every tendency evaluation.

The main function used for this data structure is `make_ode_function(model::Nonhydrostatic3DModel)`, which returns a function that computes the explicit tendency for the simulated system. If development of this interface had continued, this might have been generalized for IMEX methods by having `make_ode_function` return a `SplitODEFunction` object, which would store functions that compute the total explicit and implicit tendencies, as well as a function that computes the Jacobian of the implicit tendency.

## Why We Needed A New Interface

### 1. Modularity of Tendencies

One requirement of ClimaAtmos is that it must be easy to add new tendencies and to toggle between different versions of the same tendency. In order to add a toggle-able tendency using the *flag* paradigm, the user must

- Add a new field to the model data structure for the flag
- Define a new function with two or more methods that get dispatched based on the type of the flag
- Call the new function from `make_ode_function`, or from a function that is called by `make_ode_function`, and ensure that the flag is passed along to the new function

In other words, adding a new tendency requires expanding the model data structure and modifying the model's primary function. If development in this interface had continued, the model data structure would have ended up with dozens of different flags. Since many tendencies are governed by more than one flag, `make_ode_function` would have become a complicated mess of passing these flags to various functions that define tendencies.

The reason this issue arises is that the interface is attempting to keep tendencies modular without actually using them as first-class objects. That is, each tendency (or, more precisely, each version of a tendency) is defined in its own function method, which gives the appearance of modularity. However, the logic that determines which method gets run is hardcoded in to the model itself. For example, the model data structure stores the flag `vertical_diffusion`, and `make_ode_function` passes it to `rhs_vertical_diffusion!`, which dispatches based on this flag, as well as the `base`, `thermodynamics`, and `moisture` flags. If, instead, tendencies were first-class objects, they would be directly stored in the model data structure (as opposed to flags that govern the tendencies being stored there), and the model would simply iterate over these tendencies on every evaluation of the total tendency. Each tendency would essentially be a "black box" to the model data structure, and the logic of assembling the correct set of tendencies for the model would be handled at a higher level (i.e., by the user, or by some utility function).

Need to also mention the issue of where to store flags and parameters specific to particular tendencies.

### 2. Application of Boundary Conditions

Tendencies written with ClimaCore require boundary conditions to be evaluated +
Boundary conditions need to apply to multiple tendencies at once + Tendencies must be modular -->
*lazy evaluation of tendencies*

### 3. Cache Dependency Management

Multiple tendencies can depend on the same cached variable
Cached variables can depend on other cached variables
When a tendency term is toggled off, that might make a large number of cached variables unnecessary to evaluate
However, turning on another tendency term might make some of these cached variables necessary to evaluate again
We have only just begun seeing the difficulties of manually dealing with cache dependencies (surface temperature is an example of a cached variable that is used by multiple optional tendencies). The only way to fix this issue is through a proper dependency management system.

### 4. Integration of Non-Prognostic Variables

total surface-to-air flux for coupler
monthly means for diagnostics (e.g., temperature)

## Added Benefits of Lazy Evaluation

### 1. Automatic Jacobian Computation

Wrap input variable in `ClimaCore.Fields.jacobian_wrapper()`, then evaluate `Broadcasted` object to get derivative of tendency term with respect to input variable.

### 2. Automatic Performance Optimizations

ClimaCore operators satisfy the distributive property, so they can be rearranged for performance; e.g., $\nabla\cdot a + \nabla\cdot b + \nabla\cdot c = \nabla\cdot(a + b + c)$.
A cached variable that is only used once by a single tendency does not actually need to be cached (it can be evaluated on the fly).
Tendencies can be evaluated in parallel, if all their cached variables have already been computed.
Independent components of the cache dependency graph can be evaluated in parallel.

### 3. Automatic ClimaTimeSteppers Compatibility

In order to use ClimaTimeSteppers with limiters, it is likely that we will need to define something other than a `SplitODEFunction` that computes the total explicit and implicit tendencies. Instead, we will need a function that takes a single Euler step and applies any desired limiters. This can be generated automatically if tendencies are evaluated lazily.
