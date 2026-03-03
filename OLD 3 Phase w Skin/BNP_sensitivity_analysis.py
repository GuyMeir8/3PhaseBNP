import matplotlib
# Force non-interactive backend to avoid Tkinter errors in long loops
matplotlib.use('Agg')
import numpy as np
from configurations_3_phase import ThreePhaseConfiguration
from BNP_temperature_series_parallel_processor import BNPSeriesProcessor
from surface_energy_calculations import SurfaceEnergyValues

def run_sensitivity_analysis():
    # 1. Define Theta values to test
    # theta_values = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
    theta_values = [0.8, 0.85, 0.9, 0.95, 1.0]
    
    # 2. Instantiate base surface energies to access vacuum functions
    se_vals = SurfaceEnergyValues()
    
    # Helper to get the vacuum energy function for a specific material and phase
    def get_vac_func(mat, phase):
        return se_vals.AaBb[mat][phase]['Vacuum']

    # Helper to create the Girifalco-Good closure
    def make_gg_func(func_a, func_b, theta):
        # Returns a function of T
        return lambda T: func_a(T) + func_b(T) - 2 * theta * np.sqrt(func_a(T) * func_b(T))

    phases = ["FCC", "Liquid"]

    for theta in theta_values:
        print(f"\n========================================")
        print(f"Running Sensitivity Analysis for Theta = {theta}")
        print(f"========================================\n")
        
        overrides = {}
        
        # 3. Construct Overrides for all Ag-Cu interactions
        # We must override Ag(p1)-Cu(p2) for all combinations of phases
        for pA in phases:
            for pB in phases:
                # Get vacuum functions
                f_Ag_vac = get_vac_func("Ag", pA)
                f_Cu_vac = get_vac_func("Cu", pB)
                
                # Create new interaction function
                new_sigma_func = make_gg_func(f_Ag_vac, f_Cu_vac, theta)
                
                # Apply to both directions (Ag-Cu and Cu-Ag)
                overrides[("Ag", pA, "Cu", pB)] = new_sigma_func
                overrides[("Cu", pB, "Ag", pA)] = new_sigma_func

        # 4. Configure Simulation
        config = ThreePhaseConfiguration(
            base_file_name=f"Sensitivity_Theta_{theta}",
            plot_title_suffix=f"(Theta={theta})",
            surface_energy_overrides=overrides,
            
            # Use lower resolution for faster sensitivity testing
            xb_step=0.0125,
            t_step=12.5,
            n_total_values=[5e-17, 5e-19, 5e-21], # Test on a representative size
            t_min=500.0,
            t_max=1400.0
        )
        
        # 5. Run
        processor = BNPSeriesProcessor(config)
        processor.run(auto_show=False)

if __name__ == "__main__":
    run_sensitivity_analysis()
