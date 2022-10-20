#####
##### Utility functions
#####

is_energy_var(symbol) = symbol in (:ρθ, :ρe_tot, :ρe_int)
is_momentum_var(symbol) = symbol in (:uₕ, :ρuₕ, :w, :ρw)
is_edmf_var(symbol) = symbol in (:turbconv,)
is_tracer_var(symbol) = !(
    symbol == :ρ ||
    is_energy_var(symbol) ||
    is_momentum_var(symbol) ||
    is_edmf_var(symbol)
)
