from configurations_3_phase import low_res_configuration
from BNP_Gibbs_en_calc_3_phase import GibbsEnergyCalculator3Phase
import numpy as np

if __name__ == "__main__":
    config = low_res_configuration
    calculator = GibbsEnergyCalculator3Phase(config)    # --- Test Parameters ---
    temperatures_to_test = [800.0, 1100.0, 1400.0]
    compositions_to_test = {
        "fcc_vac": [0.01, 0.5, 0.99],
        "liq_vac": [0.01, 0.5, 0.99],
        "liq_liq": [(0.1, 0.9), (0.2, 0.8), (0.05, 0.95)],
        "sol_liq": [(0.05, 0.3), (0.1, 0.5), (0.2, 0.7)],
        "sol_sol": [(0.01, 0.99), (0.05, 0.95), (0.1, 0.9)]
    }
    primary_phases = ("FCC", "Liquid") # Needed for T_dependent_vars
    # ------------------------------------------------
    
    for T in temperatures_to_test:
        print(f"\n--- Running Tests at T = {T} K ---")
        
        # Pre-calculate T-dependent vars for this temperature
        try:
            vars = calculator._get_T_dependent_vars(T, primary_phases)
        except Exception as e:
            print(f"  Failed to get T-dependent vars. Skipping T={T}. Error: {e}")
            continue

        # 1. Alloy to Vacuum (FCC)
        print("\n  1. Alloy to Vacuum Tests:")
        for xB_fcc in compositions_to_test["fcc_vac"]:
            try:
                st_vac = calculator._calculate_surface_tension(xB_fcc, None, "FCC", None, T, primary_phases, vars)
                print(f"    ST(FCC xB={xB_fcc:.2f} -> Vacuum) @ {T}K: {st_vac:.4f} J/m^2")
            except Exception as e:
                print(f"    ST(FCC xB={xB_fcc:.2f} -> Vacuum) @ {T}K: FAILED - {e}")

        # 2. Alloy to Vacuum (Liquid)
        for xB_liq in compositions_to_test["liq_vac"]:
            try:
                st_vac = calculator._calculate_surface_tension(xB_liq, None, "Liquid", None, T, primary_phases, vars)
                print(f"    ST(Liquid xB={xB_liq:.2f} -> Vacuum) @ {T}K: {st_vac:.4f} J/m^2")
            except Exception as e:
                print(f"    ST(Liquid xB={xB_liq:.2f} -> Vacuum) @ {T}K: FAILED - {e}")

        # 3. Liquid to Liquid
        print("\n  2. Liquid to Liquid Tests:")
        for xB_l1, xB_l2 in compositions_to_test["liq_liq"]:
            try:
                st_ll = calculator._calculate_surface_tension(xB_l1, xB_l2, "Liquid", "Liquid", T, primary_phases, vars)
                print(f"    ST(Liq xB={xB_l1:.2f} -> Liq xB={xB_l2:.2f}) @ {T}K: {st_ll:.4f} J/m^2")
            except Exception as e:
                print(f"    ST(Liq xB={xB_l1:.2f} -> Liq xB={xB_l2:.2f}) @ {T}K: FAILED - {e}")

        # 4. Solid to Liquid
        print("\n  3. Solid to Liquid Tests:")
        for xB_s, xB_l in compositions_to_test["sol_liq"]:
            try:
                st_sl = calculator._calculate_surface_tension(xB_s, xB_l, "FCC", "Liquid", T, primary_phases, vars)
                print(f"    ST(FCC xB={xB_s:.2f} -> Liq xB={xB_l:.2f}) @ {T}K: {st_sl:.4f} J/m^2")
            except Exception as e:
                print(f"    ST(FCC xB={xB_s:.2f} -> Liq xB={xB_l:.2f}) @ {T}K: FAILED - {e}")

        # 5. Solid to Solid
        print("\n  4. Solid to Solid Tests:")
        for xB_s1, xB_s2 in compositions_to_test["sol_sol"]:
            try:
                st_ss = calculator._calculate_surface_tension(xB_s1, xB_s2, "FCC", "FCC", T, primary_phases, vars)
                print(f"    ST(FCC xB={xB_s1:.2f} -> FCC xB={xB_s2:.2f}) @ {T}K: {st_ss:.4f} J/m^2")
            except Exception as e:
                print(f"    ST(FCC xB={xB_s1:.2f} -> FCC xB={xB_s2:.2f}) @ {T}K: FAILED - {e}")