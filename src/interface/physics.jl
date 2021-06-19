abstract type AbstractPhysics end

# coriolis force
struct DeepShellCoriolis <: AbstractPhysics end

# gravity
struct DeepGravity <: AbstractPhysics end
struct ShallowGravity <: AbstractPhysics end