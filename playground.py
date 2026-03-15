from configurations_3_phase import low_res_configuration
from BNP_optimizer_3_phase import BNPOptimizer3Phase
import itertools
import math

if __name__ == "__main__":
    config = low_res_configuration
    optimizer = BNPOptimizer3Phase(config)

    # Define Test Ranges
    temperatures = [800, 1100, 1350]
    xBs = [0.01, 0.1, 0.5, 0.9, 0.99]
    ns = [5e-19, 5e-21]
    
    geometries = ["Janus", "Core Shell"]
    phase_combinations = [("FCC", "FCC"), ("Liquid", "Liquid"), ("FCC", "Liquid"), ("Liquid", "FCC")]
    skin_opts = [False, True]

    # Iterate through all combinations
    for T, n, xB in itertools.product(temperatures, ns, xBs):
        print(f"\nTitle: T={T} n={n} xB_total={xB}")

        for geo, phases, skin in itertools.product(geometries, phase_combinations, skin_opts):
            if skin and n == 1:
                continue

            if geo == "Janus" and phases == ("Liquid", "FCC"):
                continue

            try:
                # Using exhaustive_search=False for speed in this large test suite
                res = optimizer.find_minimum_energy(
                    T=float(T),
                    n_total=float(n),
                    xB_total=float(xB),
                    primary_phases=phases,
                    geometry_type=geo,
                    has_skin=skin,
                    exhaustive_search=True
                )
                
                # Handle potential NaNs for clean printing
                xa = res.xB_alpha if res.xB_alpha is not None else float('nan')
                xb = res.xB_beta if res.xB_beta is not None else float('nan')
                
                msg = f"  Skin={skin} Geo={geo} Phases={phases} G_total={res.G_min:.4e} xBalpha={xa:.4f} xB_beta={xb:.4f}"
                if skin:
                    msg += f", xB_skin={res.xB_skin:.4f}"
                print(msg)
                
                # Calculate and print the surface tensions for valid results
                if not math.isinf(res.G_min) and not math.isnan(xa) and not math.isnan(xb):
                    calc = optimizer.calculator
                    actual_phases, _ = calc._update_phases_based_on_skin(phases, res.xB_skin if skin else None)
                    T_dep = calc._get_T_dependent_vars(float(T), actual_phases)
                    
                    st_alpha_beta = calc._calculate_surface_tension(
                        xB_alpha=xa, xB_beta=xb,
                        phase_alpha=actual_phases[0], phase_beta=actual_phases[1],
                        T=float(T), phases=actual_phases, T_dependent_parameters=T_dep
                    )
                    
                    outer_phase = actual_phases[2] if skin else None
                    xB_outer = res.xB_skin if skin else None
                    
                    st_alpha_out = calc._calculate_surface_tension(
                        xB_alpha=xa, xB_beta=xB_outer,
                        phase_alpha=actual_phases[0], phase_beta=outer_phase,
                        T=float(T), phases=actual_phases, T_dependent_parameters=T_dep
                    )
                    
                    st_beta_out = calc._calculate_surface_tension(
                        xB_alpha=xb, xB_beta=xB_outer,
                        phase_alpha=actual_phases[1], phase_beta=outer_phase,
                        T=float(T), phases=actual_phases, T_dependent_parameters=T_dep
                    )
                    
                    print(f"    ST alpha-out={st_alpha_out:.4f} beta-out={st_beta_out:.4f} alpha-beta={st_alpha_beta:.4f}")

            except Exception as e:
                print(f"  Skin={skin} Geo={geo} Phases={phases} ERROR: {e}")

    # temperatures_to_test = [800.0, 1100.0, 1400.0]
    # compositions_to_test = {
    #     "fcc_vac": [0.01, 0.5, 0.99],
    #     "liq_vac": [0.01, 0.5, 0.99],
    #     "liq_liq": [(0.1, 0.9), (0.2, 0.8), (0.05, 0.95)],
    #     "sol_liq": [(0.05, 0.3), (0.1, 0.5), (0.2, 0.7)],
    #     "sol_sol": [(0.01, 0.99), (0.05, 0.95), (0.1, 0.9)]
    # }
    # primary_phases = ("FCC", "Liquid") # Needed for T_dependent_vars
    # # ------------------------------------------------
    
    # for T in temperatures_to_test:
    #     print(f"\n--- Running Tests at T = {T} K ---")
        
    #     # Pre-calculate T-dependent vars for this temperature
    #     try:
    #         vars = calculator._get_T_dependent_vars(T, primary_phases)
    #     except Exception as e:
    #         print(f"  Failed to get T-dependent vars. Skipping T={T}. Error: {e}")
    #         continue

    #     # 1. Alloy to Vacuum (FCC)
    #     print("\n  1. Alloy to Vacuum Tests:")
    #     for xB_fcc in compositions_to_test["fcc_vac"]:
    #         try:
    #             st_vac = calculator._calculate_surface_tension(xB_fcc, None, "FCC", None, T, primary_phases, vars)
    #             print(f"    ST(FCC xB={xB_fcc:.2f} -> Vacuum) @ {T}K: {st_vac:.4f} J/m^2")
    #         except Exception as e:
    #             print(f"    ST(FCC xB={xB_fcc:.2f} -> Vacuum) @ {T}K: FAILED - {e}")

    #     # 2. Alloy to Vacuum (Liquid)
    #     for xB_liq in compositions_to_test["liq_vac"]:
    #         try:
    #             st_vac = calculator._calculate_surface_tension(xB_liq, None, "Liquid", None, T, primary_phases, vars)
    #             print(f"    ST(Liquid xB={xB_liq:.2f} -> Vacuum) @ {T}K: {st_vac:.4f} J/m^2")
    #         except Exception as e:
    #             print(f"    ST(Liquid xB={xB_liq:.2f} -> Vacuum) @ {T}K: FAILED - {e}")

    #     # 3. Liquid to Liquid
    #     print("\n  2. Liquid to Liquid Tests:")
    #     for xB_l1, xB_l2 in compositions_to_test["liq_liq"]:
    #         try:
    #             st_ll = calculator._calculate_surface_tension(xB_l1, xB_l2, "Liquid", "Liquid", T, primary_phases, vars)
    #             print(f"    ST(Liq xB={xB_l1:.2f} -> Liq xB={xB_l2:.2f}) @ {T}K: {st_ll:.4f} J/m^2")
    #         except Exception as e:
    #             print(f"    ST(Liq xB={xB_l1:.2f} -> Liq xB={xB_l2:.2f}) @ {T}K: FAILED - {e}")

    #     # 4. Solid to Liquid
    #     print("\n  3. Solid to Liquid Tests:")
    #     for xB_s, xB_l in compositions_to_test["sol_liq"]:
    #         try:
    #             st_sl = calculator._calculate_surface_tension(xB_s, xB_l, "FCC", "Liquid", T, primary_phases, vars)
    #             print(f"    ST(FCC xB={xB_s:.2f} -> Liq xB={xB_l:.2f}) @ {T}K: {st_sl:.4f} J/m^2")
    #         except Exception as e:
    #             print(f"    ST(FCC xB={xB_s:.2f} -> Liq xB={xB_l:.2f}) @ {T}K: FAILED - {e}")

    #     # 5. Solid to Solid
    #     print("\n  4. Solid to Solid Tests:")
    #     for xB_s1, xB_s2 in compositions_to_test["sol_sol"]:
    #         try:
    #             st_ss = calculator._calculate_surface_tension(xB_s1, xB_s2, "FCC", "FCC", T, primary_phases, vars)
    #             print(f"    ST(FCC xB={xB_s1:.2f} -> FCC xB={xB_s2:.2f}) @ {T}K: {st_ss:.4f} J/m^2")
    #         except Exception as e:
    #             print(f"    ST(FCC xB={xB_s1:.2f} -> FCC xB={xB_s2:.2f}) @ {T}K: FAILED - {e}")

    #     # 6. Janus Geometry Tests
    #     print("\n  5. Janus Geometry Tests (n_total=1e-18):")
    #     n_test = 1e-18
        
    #     # Scenario A: Liquid-Liquid (Approximate complete separation 50/50)
    #     # Ag in Phase 1 (Alpha), Cu in Phase 2 (Beta)
    #     n_mp_A = np.array([[0.4*n_test, 0.1*n_test], [0.1*n_test, 0.4*n_test]]) 
    #     x_mp_A = np.array([[0.8, 0.2], [0.2, 0.8]])
    #     phases_A = ("Liquid", "Liquid")
        
    #     try:
    #         vars_A = calculator._get_T_dependent_vars(T, phases_A)
    #         skin_none = skin_class(None)
            
    #         # Calculate both to compare
    #         r_spheric = calculator._calculate_spheric_Janus_geo(n_mp_A, vars_A)
    #         r_actual = calculator._calc_Janus_geometry_for_known_nx(n_mp_A, x_mp_A, phases_A, T, vars_A, skin_none)
            
    #         sol_type = "Spheric Janus" if np.allclose(r_actual, r_spheric, rtol=1e-4) else "Regular Janus (Force Balanced)"
    #         print(f"    Liquid(Ag) / Liquid(Cu) @ {T}K: {sol_type}")
    #         print(f"      Output: r_alpha={r_actual[0]:.2e}, r_beta={r_actual[1]:.2e}, cos_theta={r_actual[2]:.4f}")
            
    #     except Exception as e:
    #         print(f"    Liquid(Ag) / Liquid(Cu) @ {T}K: FAILED - {e}")
            
    #     # Scenario B: FCC-FCC (Approximate complete separation 50/50)
    #     n_mp_B = np.array([[0.4*n_test, 0.1*n_test], [0.1*n_test, 0.4*n_test]]) 
    #     x_mp_B = np.array([[0.8, 0.2], [0.2, 0.8]])
    #     phases_B = ("FCC", "FCC")
        
    #     try:
    #         vars_B = calculator._get_T_dependent_vars(T, phases_B)
    #         skin_none = skin_class(None)
            
    #         r_spheric = calculator._calculate_spheric_Janus_geo(n_mp_B, vars_B)
    #         r_actual = calculator._calc_Janus_geometry_for_known_nx(n_mp_B, x_mp_B, phases_B, T, vars_B, skin_none)
            
    #         sol_type = "Spheric Janus" if np.allclose(r_actual, r_spheric, rtol=1e-4) else "Regular Janus (Force Balanced)"
    #         print(f"    FCC(Ag) / FCC(Cu) @ {T}K: {sol_type}")
    #         print(f"      Output: r_alpha={r_actual[0]:.2e}, r_beta={r_actual[1]:.2e}, cos_theta={r_actual[2]:.4f}")
    #     except Exception as e:
    #          print(f"    FCC(Ag) / FCC(Cu) @ {T}K: FAILED - {e}")