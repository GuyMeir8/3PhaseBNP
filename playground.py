from configurations_3_phase import low_res_configuration
from BNP_Gibbs_en_calc_3_phase import GibbsEnergyCalculator3Phase
import numpy as np

if __name__ == "__main__":
    config = low_res_configuration
    calculator = GibbsEnergyCalculator3Phase(config)
    
    # --- Adjustable Inputs for Energy Calculation ---
    A_ratio_alpha = 0.5
    B_ratio_alpha = 0.5
    T = 1000.0
    n_total = 1e-18
    xB_total = 0.4
    primary_phases = ("FCC", "Liquid")
    geometry_type = "Janus"
    skin_val = None
    # ------------------------------------------------
    
    # Pre-calculate vars
    vars = calculator._get_T_dependent_vars(T, primary_phases)
    
    # print(f"--- Full Energy Calculation Test (T={T} K) ---")
    # calculator.calculate_total_energy(
    #     A_ratio_alpha=A_ratio_alpha,
    #     B_ratio_alpha=B_ratio_alpha,
    #     T=T,
    #     n_total=n_total,
    #     xB_total=xB_total,
    #     primary_phases=primary_phases,
    #     geometry_type=geometry_type,
    #     skin_val=skin_val
    # )
    
    print(f"--- Surface Tension Unit Tests (T={T} K) ---")
    
    # 1. Alloy to Vacuum (FCC)
    xB_fcc = 0.1
    try:
        st_vac = calculator._calculate_surface_tension(xB_fcc, None, "Liquid", None, T, primary_phases, vars)
        print(f"ST(FCC {xB_fcc} -> Vacuum): {st_vac:.4f} J/m^2")
    except Exception as e:
        print(f"ST(FCC -> Vacuum) Failed: {e}")

    # 2. Liquid to Liquid (Miscibility Gap check, though Ag-Cu is eutectic, let's just test math)
    xB_l1 = 0.2
    xB_l2 = 0.8
    try:
        st_ll = calculator._calculate_surface_tension(xB_l1, xB_l2, "Liquid", "Liquid", T, primary_phases, vars)
        print(f"ST(Liquid {xB_l1} -> Liquid {xB_l2}): {st_ll:.4f} J/m^2")
    except Exception as e:
        print(f"ST(Liquid -> Liquid) Failed: {e}")

    # 3. Solid to Liquid
    xB_s = 0.1
    xB_l = 0.4
    try:
        st_sl = calculator._calculate_surface_tension(xB_s, xB_l, "FCC", "Liquid", T, primary_phases, vars)
        print(f"ST(FCC {xB_s} -> Liquid {xB_l}): {st_sl:.4f} J/m^2")
    except Exception as e:
        print(f"ST(FCC -> Liquid) Failed: {e}")

    # 4. Solid to Solid (Grain Boundary / Phase Boundary)
    xB_s1 = 0.05
    xB_s2 = 0.95
    try:
        st_ss = calculator._calculate_surface_tension(xB_s1, xB_s2, "FCC", "FCC", T, primary_phases, vars)
        print(f"ST(FCC {xB_s1} -> FCC {xB_s2}): {st_ss:.4f} J/m^2")
    except Exception as e:
        print(f"ST(FCC -> FCC) Failed: {e}")