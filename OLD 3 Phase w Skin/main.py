import glob
import os
from configurations_3_phase import ThreePhaseConfiguration, low_res_configuration
from system_data import SystemData
from BNP_Gibbs_en_calc_3_phase import GibbsEnergyCalculator3Phase
from BNP_optimizer_3_phase import BNPOptimizer3Phase
from plotting_3_phase import PhaseDiagramPlotting3Phase

if __name__ == "__main__":
    config = ThreePhaseConfiguration(
        base_file_name="3Phase_main_test",
        xb_step=0.01,
        t_step=10.0,
        n_total_values=[1, 5e-17, 5e-19, 5e-21],
        t_min=500.0,
    )
    system_data = SystemData(config)
    optimizer = BNPOptimizer3Phase(system_data)

    test_res = optimizer.calculator.calculate_total_energy(
        n_total=5e-17,
        A_ratio_alpha=0.0001,
        B_ratio_alpha=0.0001,
        T=400.0,
        xB_total=0.01,
        primary_phases=("FCC", "FCC"),
        geometry_type="Janus",
        has_skin=True,
        xB_skin=0.0001,
    )
    
    # print("Test Result:", test_res)

    # Automatically find and plot the latest result file in the Results folder
    list_of_files = glob.glob('Results/*.csv') 
    if list_of_files:
        latest_file = max(list_of_files, key=os.path.getctime)
        print(f"Found latest file: {latest_file}")
        PhaseDiagramPlotting3Phase(latest_file)
    else:
        print("No result files found in Results/ directory.")