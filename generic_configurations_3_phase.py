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
