from configurations_3_phase import ThreePhaseConfiguration, low_res_configuration
from system_data import SystemData
from BNP_Gibbs_en_calc_3_phase import GibbsEnergyCalculator3Phase
if __name__ == "__main__":
    config = ThreePhaseConfiguration(
        base_file_name="3Phase_main_test",
        xb_step=0.01,
        t_step=10.0,
        n_total_values=[1, 5e-17, 5e-19, 5e-21],
        t_min=500.0,
    )
    system_data = SystemData(config)
    test_calc = GibbsEnergyCalculator3Phase(system_data)
    test_res = test_calc.calculate_total_energy(
        A_ratio_alpha=0.5,
        B_ratio_alpha=0.4,
        n_total=5e-17,
        T=700.0,
        xB_total=0.9,
        primary_phases=("FCC", "FCC"),
        geometry_type="Core_Shell",
        has_skin=False,
        xB_skin=0.2
    )
        
    print("Test Calculation Result:", test_res)