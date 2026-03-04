from configurations_3_phase import low_res_configuration
from BNP_Gibbs_en_calc_3_phase import GibbsEnergyCalculator3Phase

if __name__ == "__main__":
    # --- Adjustable Inputs ---
    A_ratio_alpha = 0.3
    B_ratio_alpha = 0.5
    T = 800.0
    n_total = 1e-18
    xB_total = 0.4
    primary_phases = ("FCC", "Liquid")
    geometry_type = "Janus"
    skin_val = None  # Must be None to enter the no-skin Janus path
    # -------------------------

    config = low_res_configuration
    calculator = GibbsEnergyCalculator3Phase(config)
    
    print(f"Running calculation for {geometry_type} at T={T}...")
    
    calculator.calculate_total_energy(
        A_ratio_alpha=A_ratio_alpha,
        B_ratio_alpha=B_ratio_alpha,
        T=T,
        n_total=n_total,
        xB_total=xB_total,
        primary_phases=primary_phases,
        geometry_type=geometry_type,
        skin_val=skin_val
    )
    
    print("Done.")