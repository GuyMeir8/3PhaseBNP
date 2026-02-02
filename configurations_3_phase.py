from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Union, Callable, Optional
import numpy as np

@dataclass
class ThreePhaseConfiguration:
    """
    Configuration for the 3-Phase BNP simulation.
    
    Attributes:
        base_file_name: Prefix for output files.
        materials: List of material names (e.g., ["Ag", "Cu"]).
        phases: List of available phases (e.g., ["FCC", "Liquid"]).
        geometries: List of available geometries (e.g., ["Core_Shell", "Janus"]).
        
        surface_energy_overrides: 
            A dictionary to override specific surface energy values from the database.
            Key: A tuple representing the path to the value.
                 e.g. ("Ag", "FCC", "Cu", "Liquid") or ("Ag", "FCC", "Vacuum")
            Value: Either a float (constant energy) or a function (T -> energy).
    """
    base_file_name: str = "3Phase_Simulation"
    
    # System Definition
    materials: Tuple[str, str] = ("Ag", "Cu")
    phases: Tuple[str, ...] = ("FCC", "Liquid")
    geometries: Tuple[str, ...] = ("Core_Shell", "Janus")
    
    # Sensitivity Analysis / Overrides
    # Example: { ("Ag", "FCC", "Cu", "Liquid"): 0.15 }
    surface_energy_overrides: Dict[Tuple[str, ...], Union[float, Callable[..., float]]] = field(default_factory=dict)

    # Simulation Grid
    xb_step: float = 0.01
    t_min: float = 400.0
    t_max: float = 1400.0
    t_step: float = 10.0
    n_total_values: List[float] = field(default_factory=lambda: [1.0, 5e-17, 5e-19, 5e-21])

    @property
    def xb_values(self) -> List[float]:
        """Generates xB (composition) values."""
        epsilon = self.xb_step / 1000.0
        values = np.arange(self.xb_step, 1.0 - self.xb_step + epsilon, self.xb_step)
        return [float(round(x, 6)) for x in values]

    @property
    def temperature_values(self) -> List[float]:
        """Generates Temperature values."""
        epsilon = self.t_step / 1000.0
        values = np.arange(self.t_min, self.t_max + epsilon, self.t_step)
        return [float(round(x, 6)) for x in values]

# --- Pre-defined Configuration Instances ---

# 1. Standard / High Resolution (Default)
standard_configuration = ThreePhaseConfiguration()

# 2. Low Resolution (Faster, good for general trends)
low_res_configuration = ThreePhaseConfiguration(
    base_file_name="3Phase_LowRes",
    xb_step=0.01,
    t_step=10.0,
    n_total_values=[1, 5e-17, 5e-19, 5e-21]
    # n_total_values=[1, 5e-17]
)

# 3. Debugging (Very fast, minimal points to check if code runs)
debug_configuration = ThreePhaseConfiguration(
    base_file_name="3Phase_Debug",
    xb_step=0.2,
    t_step=200.0,
    n_total_values=[5e-19]
)

# 4. Sensitivity Example (Override specific surface energy)
sensitivity_configuration = ThreePhaseConfiguration(
    base_file_name="3Phase_Sensitivity_AgCu_Liquid",
    surface_energy_overrides={
        ("Ag", "FCC", "Cu", "Liquid"): 0.5  # Example: Force this interface energy to 0.5
    }
)
