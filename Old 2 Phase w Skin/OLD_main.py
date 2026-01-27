from plotting_file import PhaseDiagramPlotting
from generic_configuration_gibbs_free_energy import low_res_configuration
from Gibbs_energy_calculator import GibbsEnergyCalculator
from Gibbs_free_energy_optimizer import GibbsOptimizer
from generic_configuration_gibbs_free_energy import GeometrySettings, GeometryType, GenericConfiguration, ShellSettings

if __name__ == "__main__":
    # 1. Initialize configuration (low_res_configuration is already initialized in its module)
    config = low_res_configuration

    # 2. Insert configuration into Gibbs_energy_calculator
    calculator = GibbsEnergyCalculator(config)

    optimizer = GibbsOptimizer(low_res_configuration)

    # sol = calculator.calculate_single_phase_gibbs_free_energy(
    #     n_total=5e-21,
    #     xB_total=0.01,
    #     T=700.0,
    #     phase="FCC",
    #     geometry=config.geometry_options[7]
    # )
    #
    # geometry_val = GeometrySettings(
    #         geometry_type=GeometryType.SPHERIC_JANUS,
    #         has_outer_shell=True,
    #         shell=ShellSettings(material=0, phase="Liquid")
    #     )
    # sol = optimizer.solve_specific_configuration(
    #     T=700.0,
    #     xB_total=0.1,
    #     n_total=5e-17,
    #     geometry=geometry_val,
    #     phase_pair=("FCC", "FCC"),
    #     initial_guess= [0.5, 0.5]
    # )
    #
    # print(sol)
    PhaseDiagramPlotting("Results\low_res_FCC_like_Cu_Liquid_like_Ag_20260125_181056.csv")




