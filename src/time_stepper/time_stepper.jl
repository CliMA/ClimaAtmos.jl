import NVTX
import ClimaTimeSteppers as CTS
import Krylov

Base.@kwdef struct AtmosODEFunction{TL, TE, TI, L, D, PE, PI} <:
                   CTS.AbstractClimaODEFunction
    T_lim!::TL = (uₜ, u, p, t) -> nothing
    T_exp!::TE = (uₜ, u, p, t) -> nothing
    T_imp!::TI = (uₜ, u, p, t) -> nothing
    lim!::L = (u, p, t, u_ref) -> nothing
    dss!::D = (u, p, t) -> nothing
    post_explicit!::PE = (u, p, t) -> nothing
    post_implicit!::PI = (u, p, t) -> nothing
end

include("imex_ark.jl")
include("imex_ssprk.jl")
include("hc_ars343.jl")
