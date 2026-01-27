import inspect
from typing import Dict, Any, Callable, Union, Type
import pure_material_properties
import interaction_properties
from pure_material_properties import BaseMaterial
from surface_energy_calculations import SurfaceEnergyValues
from interaction_properties import BaseInteraction
from generic_configurations_3_phase import ThreePhaseConfiguration


class SystemData:
    """
    Central data registry for the simulation.
    Loads standard database values and applies any overrides from the configuration.
    """
    def __init__(self, config: ThreePhaseConfiguration):
        self.config = config
        
        # 1. Dynamically load all materials and interactions from their respective modules.
        self.material_data = self._load_data_from_module(pure_material_properties, BaseMaterial)
        self.interaction_data = self._load_data_from_module(interaction_properties, BaseInteraction)
        
        self.surface_energy = SurfaceEnergyValues()
        
        # 2. Apply Sensitivity Overrides
        self._apply_surface_overrides()

    def _load_data_from_module(self, module: Any, base_class: Type) -> Dict[str, Any]:
        """
        Dynamically loads all data classes from a given module that inherit from a base_class.
        It finds all classes in the file, instantiates them, and stores them in a dictionary
        keyed by their 'name' attribute.
        """
        data_dict = {}
        for _, obj in inspect.getmembers(module):
            # Ensure the object is a class, a subclass of our target, but not the target itself.
            if inspect.isclass(obj) and issubclass(obj, base_class) and obj is not base_class:
                instance = obj()
                data_dict[instance.name] = instance
        return data_dict

    def _apply_surface_overrides(self):
        """
        Patches the SurfaceEnergyValues object with overrides from config.
        """
        if not self.config.surface_energy_overrides:
            return

        for path_tuple, value in self.config.surface_energy_overrides.items():
            # Ensure value is a callable (function of T and/or others)
            final_func = value
            if not callable(value):
                # Capture value in a closure to avoid late binding issues
                final_func = (lambda v=value: lambda **kwargs: v)()
            
            # Navigate the nested dictionary structure
            # Structure is: self.surface_energy.AaBb[MatA][PhaseA][MatB][PhaseB]
            current_dict: Any = self.surface_energy.AaBb
            
            # Traverse down to the second-to-last key
            for key in path_tuple[:-1]:
                current_dict = current_dict[key]
            
            # Set the value at the final key
            last_key = path_tuple[-1]
            current_dict[last_key] = final_func



    def get_material(self, name: str) -> BaseMaterial:
        return self.material_data[name]

    def get_interaction(self, name: str) -> BaseInteraction:
        return self.interaction_data[name]
