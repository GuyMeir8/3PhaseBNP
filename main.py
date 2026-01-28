from configurations_3_phase import ThreePhaseConfiguration, low_res_configuration
from system_data import SystemData
if __name__ == "__main__":
    config = ThreePhaseConfiguration(
        base_file_name="3Phase_main_test",
        xb_step=0.01,
        t_step=10.0,
        n_total_values=[1, 5e-17, 5e-19, 5e-21],
        t_min=500.0,
    )
    system_data = SystemData(config)