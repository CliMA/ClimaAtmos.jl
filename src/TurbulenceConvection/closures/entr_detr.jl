function compute_entr_detr!(
    state::State,
    grid::Grid,
    edmf::EDMFModel,
    param_set::APS,
)
    FT = float_type(state)

    N_up = n_updrafts(edmf)
    aux_up = center_aux_updrafts(state)
    aux_en = center_aux_environment(state)
    aux_gm = center_aux_grid_mean(state)

    @inbounds for i in 1:N_up
        @inbounds for k in real_center_indices(grid)
            if aux_up[i].area[k] > 0.0

                a_en = aux_en.area[k] # environment area fraction
                tke_gm = aux_gm.tke[k] # grid-mean tke
                tke_en = aux_en.tke[k] # environment tke
                Π_2 = (tke_gm - a_en * tke_en) / (tke_gm + eps(FT))

                RH_up = aux_up[i].RH[k] # updraft relative humidity
                RH_en = aux_en.RH[k] # environment relative humidity
                Π_4 = RH_up - RH_en

                aux_up[i].detr_sc[k] =
                    max(-0.0102 + 0.0612 * Π_2 + 0.0827 * Π_4, FT(0))
            else
                aux_up[i].detr_sc[k] = FT(0)
            end

            aux_up[i].entr_sc[k] = FT(5.0e-4)
            aux_up[i].frac_turb_entr[k] = FT(0)

            aux_up[i].entr_turb_dyn[k] =
                aux_up[i].entr_sc[k] + aux_up[i].frac_turb_entr[k]
            aux_up[i].detr_turb_dyn[k] =
                aux_up[i].detr_sc[k] + aux_up[i].frac_turb_entr[k]
        end
    end
end
